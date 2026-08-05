"""Shared loading, artifact, and ONNX contract helpers."""

import hashlib
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from flax import serialization


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "training" / "model"
FEATURES_DIR = REPO_ROOT / "training" / "features"
for import_dir in (MODEL_DIR, FEATURES_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from architecture import ChatClassifier
from preprocessing import FEATURE_NAMES


MODEL_FILENAME = "chat_classifier.onnx"
VOCAB_FILENAME = "vocab.json"
META_FILENAME = "inference_meta.json"
PARAMS_FILENAME = "chat_classifier_params.msgpack"
MANIFEST_FILENAME = "manifest.json"
INPUT_NAMES = ("tokens", "features")
OUTPUT_NAMES = ("logits",)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_artifacts(run_dir):
    run_dir = Path(run_dir)
    required = (PARAMS_FILENAME, VOCAB_FILENAME, META_FILENAME)
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run directory {run_dir} is missing: {', '.join(missing)}"
        )

    metadata = load_json(run_dir / META_FILENAME)
    vocab = load_json(run_dir / VOCAB_FILENAME)
    required_meta = (
        "vocab_size",
        "embed_dim",
        "hidden_dim",
        "max_seq_len",
        "num_features",
        "feature_names",
        "feature_mean",
        "feature_std",
        "stream_time_scale_seconds",
        "threshold",
        "split",
        "seed",
    )
    missing_meta = [name for name in required_meta if name not in metadata]
    if missing_meta:
        raise ValueError(
            f"{META_FILENAME} is missing fields: {', '.join(missing_meta)}"
        )

    vocab_size = int(metadata["vocab_size"])
    num_features = int(metadata["num_features"])
    if len(vocab) != vocab_size:
        raise ValueError(
            f"Vocabulary size mismatch: metadata={vocab_size}, file={len(vocab)}"
        )
    for field in ("feature_names", "feature_mean", "feature_std"):
        if len(metadata[field]) != num_features:
            raise ValueError(
                f"{field} has {len(metadata[field])} entries; expected {num_features}"
            )
    token_ids = sorted(int(token_id) for token_id in vocab.values())
    if token_ids != list(range(vocab_size)):
        raise ValueError("Vocabulary IDs must be contiguous from 0 to vocab_size - 1.")
    expected_special_tokens = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2}
    for token, expected_id in expected_special_tokens.items():
        if vocab.get(token) != expected_id:
            raise ValueError(
                f"Vocabulary must map {token} to {expected_id}; got {vocab.get(token)!r}."
            )
    if metadata["feature_names"] != FEATURE_NAMES:
        raise ValueError(
            "Saved feature_names do not match training/features/preprocessing.py."
        )
    return metadata, vocab


def load_jax_model(run_dir, metadata):
    """Recreate the production architecture and load its saved variable tree."""
    model = ChatClassifier(
        vocab_size=int(metadata["vocab_size"]),
        embed_dim=int(metadata["embed_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        num_features=int(metadata["num_features"]),
    )
    dummy_tokens = jnp.zeros(
        (1, int(metadata["max_seq_len"])),
        dtype=jnp.int32,
    )
    dummy_features = jnp.zeros(
        (1, int(metadata["num_features"])),
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(0)
    template = model.init(
        {"params": key, "dropout": key},
        dummy_tokens,
        dummy_features,
        training=False,
    )
    with open(Path(run_dir) / PARAMS_FILENAME, "rb") as file:
        variables = serialization.from_bytes(template, file.read())
    variables = jax.tree_util.tree_map(jnp.asarray, variables)
    return model, variables


def export_forward(variables, tokens, features):
    """Inference-only forward pass that avoids Flax's lifted RNN tracing.

    jax2onnx currently turns a shape inside flax.linen.RNN/axes_scan into a
    JitTracer. These equations are the exact flax.linen.GRUCell equations used
    by ChatClassifier, applied to the same saved parameter tree.
    """
    params = variables["params"]
    embedded = jnp.take(params["embed"]["embedding"], tokens, axis=0)
    gru = params["GRUCell_0"]

    def dense(x, layer, use_bias=True):
        output = jnp.matmul(x, layer["kernel"])
        if use_bias:
            output = output + layer["bias"]
        return output

    def step(hidden, inputs):
        reset = jax.nn.sigmoid(
            dense(inputs, gru["ir"]) + dense(hidden, gru["hr"], use_bias=False)
        )
        update = jax.nn.sigmoid(
            dense(inputs, gru["iz"]) + dense(hidden, gru["hz"], use_bias=False)
        )
        candidate = jnp.tanh(
            dense(inputs, gru["in"]) + reset * dense(hidden, gru["hn"])
        )
        new_hidden = (1.0 - update) * candidate + update * hidden
        return new_hidden, new_hidden

    initial_hidden = jnp.zeros(
        (embedded.shape[0], gru["hn"]["bias"].shape[0]),
        dtype=embedded.dtype,
    )
    final_hidden, _ = jax.lax.scan(
        step,
        initial_hidden,
        jnp.swapaxes(embedded, 0, 1),
    )

    combined = jnp.concatenate([final_hidden, features], axis=-1)
    head = jax.nn.relu(dense(combined, params["dense_1"]))
    logits = dense(head, params["dense_out"])
    return logits.squeeze(-1)


def validate_export_forward(model, variables, metadata):
    """Prove the export-only equations match the original Flax module."""
    rng = np.random.default_rng(0)
    samples = [
        (
            np.zeros((1, int(metadata["max_seq_len"])), dtype=np.int32),
            np.zeros((1, int(metadata["num_features"])), dtype=np.float32),
        ),
        (
            rng.integers(
                0,
                int(metadata["vocab_size"]),
                size=(1, int(metadata["max_seq_len"])),
                dtype=np.int32,
            ),
            rng.normal(
                size=(1, int(metadata["num_features"])),
            ).astype(np.float32),
        ),
    ]
    for tokens, features in samples:
        flax_logits = np.asarray(
            model.apply(
                variables,
                jnp.asarray(tokens),
                jnp.asarray(features),
                training=False,
            )
        )
        export_logits = np.asarray(
            export_forward(
                variables,
                jnp.asarray(tokens),
                jnp.asarray(features),
            )
        )
        np.testing.assert_allclose(
            export_logits,
            flax_logits,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Export-only GRU equations differ from the Flax model.",
        )


def validate_onnx_structure(model_path, metadata, run_smoke=True):
    """Run checker, inferred-shape checks, runtime load, and contract validation."""
    model_path = Path(model_path)
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if [item.name for item in inputs] != list(INPUT_NAMES):
        raise ValueError(
            f"ONNX inputs must be {INPUT_NAMES}; got {[item.name for item in inputs]}"
        )
    if [item.name for item in outputs] != list(OUTPUT_NAMES):
        raise ValueError(
            f"ONNX outputs must be {OUTPUT_NAMES}; got {[item.name for item in outputs]}"
        )

    expected = {
        "tokens": ("tensor(int32)", [1, int(metadata["max_seq_len"])]),
        "features": ("tensor(float)", [1, int(metadata["num_features"])]),
        "logits": ("tensor(float)", [1]),
    }
    for item in (*inputs, *outputs):
        expected_type, expected_shape = expected[item.name]
        if item.type != expected_type:
            raise TypeError(
                f"{item.name} type must be {expected_type}; got {item.type}"
            )
        if item.shape != expected_shape:
            raise ValueError(
                f"{item.name} shape must be {expected_shape}; got {item.shape}"
            )

    if run_smoke:
        result = session.run(
            list(OUTPUT_NAMES),
            {
                "tokens": np.zeros(expected["tokens"][1], dtype=np.int32),
                "features": np.zeros(expected["features"][1], dtype=np.float32),
            },
        )
        logits = np.asarray(result[0])
        if logits.shape != (1,) or not np.all(np.isfinite(logits)):
            raise ValueError(
                f"Runtime smoke output must be one finite logit; got {logits}"
            )
    return session


def verify_manifest(export_dir, run_dir=None):
    export_dir = Path(export_dir)
    manifest_path = export_dir / MANIFEST_FILENAME
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported manifest schema: {manifest.get('schema_version')!r}"
        )

    expected_artifacts = {MODEL_FILENAME, VOCAB_FILENAME, META_FILENAME}
    artifact_checksums = manifest.get("artifacts")
    if not isinstance(artifact_checksums, dict) or set(artifact_checksums) != expected_artifacts:
        raise ValueError(
            f"Manifest artifacts must be exactly {sorted(expected_artifacts)}."
        )
    for name, expected_digest in artifact_checksums.items():
        path = export_dir / name
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Checksum mismatch for {path}: "
                f"expected {expected_digest}, got {actual_digest}"
            )

    if run_dir is not None:
        run_dir = Path(run_dir)
        expected_sources = {PARAMS_FILENAME, VOCAB_FILENAME, META_FILENAME}
        source_checksums = manifest.get("source_artifacts")
        if not isinstance(source_checksums, dict) or set(source_checksums) != expected_sources:
            raise ValueError(
                f"Manifest source artifacts must be exactly {sorted(expected_sources)}."
            )
        for name, expected_digest in source_checksums.items():
            path = run_dir / name
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"Source checksum mismatch for {path}: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
    return manifest


def display_path(path):
    try:
        return os.path.relpath(Path(path), REPO_ROOT)
    except ValueError:
        return str(path)
