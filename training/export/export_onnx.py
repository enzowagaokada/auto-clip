"""Export a saved Flax production run directly to an ONNX deployment bundle.

Run from the repository root:
    python training/export/export_onnx.py
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from jax2onnx import to_onnx

from export_common import (
    INPUT_NAMES,
    MANIFEST_FILENAME,
    META_FILENAME,
    MODEL_FILENAME,
    OUTPUT_NAMES,
    PARAMS_FILENAME,
    VOCAB_FILENAME,
    display_path,
    export_forward,
    load_jax_model,
    sha256_file,
    validate_export_forward,
    validate_onnx_structure,
    validate_run_artifacts,
)


DEFAULT_RUN_DIR = Path("models/runs/reviewed-vod-seed0")
DEFAULT_OUTPUT_DIR = Path("models/exports/reviewed-vod-seed0")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a saved JAX/Flax chat classifier directly to ONNX."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--opset",
        type=int,
        default=23,
        help="ONNX opset passed to jax2onnx (default: 23).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing deployment bundle artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata, _ = validate_run_artifacts(args.run_dir)
    model, variables = load_jax_model(args.run_dir, metadata)
    validate_export_forward(model, variables, metadata)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        MODEL_FILENAME,
        VOCAB_FILENAME,
        META_FILENAME,
        MANIFEST_FILENAME,
    )
    existing = [output_dir / name for name in artifact_names if (output_dir / name).exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Export artifacts already exist: {paths}. Pass --overwrite to replace them."
        )
    for path in existing:
        path.unlink()

    def inference(tokens, features):
        return export_forward(
            variables,
            tokens,
            features,
        )

    model_path = output_dir / MODEL_FILENAME
    dummy_tokens = np.zeros(
        (1, int(metadata["max_seq_len"])),
        dtype=np.int32,
    )
    dummy_features = np.zeros(
        (1, int(metadata["num_features"])),
        dtype=np.float32,
    )
    to_onnx(
        inference,
        inputs=[dummy_tokens, dummy_features],
        model_name="twitch_chat_classifier",
        opset=args.opset,
        return_mode="file",
        output_path=str(model_path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
    )

    # The deployment contract is intentionally batch-one for the live Go loop.
    validate_onnx_structure(model_path, metadata, run_smoke=True)

    shutil.copy2(args.run_dir / VOCAB_FILENAME, output_dir / VOCAB_FILENAME)
    shutil.copy2(args.run_dir / META_FILENAME, output_dir / META_FILENAME)

    artifact_checksums = {
        name: sha256_file(output_dir / name)
        for name in (MODEL_FILENAME, VOCAB_FILENAME, META_FILENAME)
    }
    source_checksums = {
        name: sha256_file(args.run_dir / name)
        for name in (PARAMS_FILENAME, VOCAB_FILENAME, META_FILENAME)
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": display_path(args.run_dir),
        "model_format": "ONNX",
        "exporter": "jax2onnx",
        "forward_implementation": "explicit_flax_linen_gru_equations",
        "opset": args.opset,
        "contract": {
            "inputs": {
                "tokens": {
                    "dtype": "int32",
                    "shape": [1, int(metadata["max_seq_len"])],
                },
                "features": {
                    "dtype": "float32",
                    "shape": [1, int(metadata["num_features"])],
                    "names": metadata["feature_names"],
                },
            },
            "outputs": {
                "logits": {
                    "dtype": "float32",
                    "shape": [1],
                    "activation": "none",
                }
            },
        },
        "artifacts": artifact_checksums,
        "source_artifacts": source_checksums,
        "versions": {
            name: package_version(name)
            for name in ("jax", "flax", "jax2onnx", "onnx", "onnxruntime")
        },
    }
    with open(output_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Exported ONNX bundle to {display_path(output_dir)}")
    print(
        "Contract: "
        f"tokens[int32, 1x{metadata['max_seq_len']}], "
        f"features[float32, 1x{metadata['num_features']}] "
        "-> logits[float32, 1]"
    )
    print(f"Manifest: {display_path(output_dir / MANIFEST_FILENAME)}")


if __name__ == "__main__":
    main()
