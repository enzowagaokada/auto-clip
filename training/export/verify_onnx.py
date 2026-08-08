"""Validate an ONNX bundle and compare it with JAX on saved validation rows.

Run from the repository root:
    python training/export/verify_onnx.py
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from export_common import (
    META_FILENAME,
    MODEL_FILENAME,
    VOCAB_FILENAME,
    display_path,
    load_jax_model,
    load_json,
    validate_onnx_structure,
    validate_run_artifacts,
    verify_manifest,
)

from data import (
    load_dataset_rows,
    prepare_dataset_from_saved_preprocessing,
    streamer_holdout_split,
    vod_group_split,
    vod_manifest_split,
)


DEFAULT_RUN_DIR = Path("models/runs/window-v2-vod-seed0")
DEFAULT_EXPORT_DIR = Path("models/exports/window-v2-vod-seed0")
DEFAULT_DATASET = Path("data/processed/dataset.jsonl")
DEFAULT_CONFIG = Path("config.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check an ONNX deployment bundle and compare logits with the saved "
            "JAX model on its reconstructed validation rows."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit parity rows for a quick check; 0 verifies every validation row.",
    )
    parser.add_argument("--jax-batch-size", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def reconstruct_validation_indices(rows, metadata, config):
    if metadata.get("holdout_vod_ids"):
        _, validation = vod_manifest_split(rows, metadata["holdout_vod_ids"])
    else:
        split = metadata["split"]
        if split.startswith("holdout streamer ="):
            streamer = split.partition("=")[2].strip()
            _, validation = streamer_holdout_split(rows, streamer)
        elif split.startswith("VOD-grouped"):
            validation_fraction = float(
                config["training"].get("validation_fraction", 0.2)
            )
            _, validation = vod_group_split(
                rows,
                val_frac=validation_fraction,
                seed=int(metadata.get("seed", 0)),
            )
        else:
            raise ValueError(
                f"Cannot reconstruct unsupported saved split {split!r}."
            )

    saved_confusion = metadata.get("best_val_metrics", {}).get("confusion", {})
    expected_size = sum(
        int(saved_confusion.get(name, 0)) for name in ("tp", "fp", "fn", "tn")
    )
    return np.asarray(validation, dtype=np.int64), expected_size


def jax_logits(model, variables, tokens, features, batch_size):
    batches = []
    for start in range(0, len(tokens), batch_size):
        end = start + batch_size
        batch = model.apply(
            variables,
            jnp.asarray(tokens[start:end]),
            jnp.asarray(features[start:end]),
            training=False,
        )
        batches.append(np.asarray(batch, dtype=np.float32))
    return np.concatenate(batches)


def onnx_logits(session, tokens, features):
    predictions = np.empty(len(tokens), dtype=np.float32)
    for index in range(len(tokens)):
        result = session.run(
            ["logits"],
            {
                "tokens": np.ascontiguousarray(tokens[index : index + 1], dtype=np.int32),
                "features": np.ascontiguousarray(
                    features[index : index + 1],
                    dtype=np.float32,
                ),
            },
        )
        predictions[index] = np.asarray(result[0], dtype=np.float32).reshape(-1)[0]
    return predictions


def main():
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows must be zero or positive.")
    if args.jax_batch_size <= 0:
        raise ValueError("--jax-batch-size must be positive.")

    run_metadata, run_vocab = validate_run_artifacts(args.run_dir)
    verify_manifest(args.export_dir, run_dir=args.run_dir)

    exported_metadata = load_json(args.export_dir / META_FILENAME)
    exported_vocab = load_json(args.export_dir / VOCAB_FILENAME)
    if exported_metadata != run_metadata:
        raise ValueError("Exported inference_meta.json differs from the source run.")
    if exported_vocab != run_vocab:
        raise ValueError("Exported vocab.json differs from the source run.")

    session = validate_onnx_structure(
        args.export_dir / MODEL_FILENAME,
        run_metadata,
        run_smoke=True,
    )
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    rows = load_dataset_rows(str(args.dataset))
    validation_indices, saved_validation_size = reconstruct_validation_indices(
        rows,
        run_metadata,
        config,
    )
    total_validation_rows = len(validation_indices)
    matches_saved_validation = (
        not saved_validation_size or total_validation_rows == saved_validation_size
    )
    if not matches_saved_validation:
        print(
            "Warning: the processed dataset has changed since training, so this "
            "is a numerical parity sample rather than a reproduction of the "
            "saved validation metrics "
            f"(saved={saved_validation_size}, current selection={total_validation_rows})."
        )
    if args.max_rows:
        validation_indices = validation_indices[: args.max_rows]
    if not len(validation_indices):
        raise ValueError("No validation rows were selected for parity verification.")

    tokens, features, _ = prepare_dataset_from_saved_preprocessing(
        rows=rows,
        vocab=run_vocab,
        max_seq_len=int(run_metadata["max_seq_len"]),
        stream_time_scale_seconds=int(run_metadata["stream_time_scale_seconds"]),
        feature_mean=run_metadata["feature_mean"],
        feature_std=run_metadata["feature_std"],
    )
    selected_tokens = tokens[validation_indices]
    selected_features = features[validation_indices]

    model, variables = load_jax_model(args.run_dir, run_metadata)
    expected = jax_logits(
        model,
        variables,
        selected_tokens,
        selected_features,
        args.jax_batch_size,
    )
    actual = onnx_logits(session, selected_tokens, selected_features)
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise ValueError("JAX or ONNX produced non-finite validation logits.")

    absolute_error = np.abs(expected - actual)
    relative_error = absolute_error / np.maximum(np.abs(expected), args.atol)
    close = np.isclose(expected, actual, atol=args.atol, rtol=args.rtol)
    expected_scores = 1.0 / (1.0 + np.exp(-expected))
    actual_scores = 1.0 / (1.0 + np.exp(-actual))
    threshold = float(run_metadata["threshold"])
    decision_mismatches = np.count_nonzero(
        (expected_scores >= threshold) != (actual_scores >= threshold)
    )
    worst_index = int(np.argmax(absolute_error))
    summary = {
        "source_run": display_path(args.run_dir),
        "onnx_model": display_path(args.export_dir / MODEL_FILENAME),
        "saved_validation_rows": saved_validation_size,
        "current_selection_rows": total_validation_rows,
        "selection_matches_saved_validation": matches_saved_validation,
        "rows_compared": len(validation_indices),
        "atol": args.atol,
        "rtol": args.rtol,
        "max_absolute_error": float(absolute_error[worst_index]),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "max_relative_error": float(np.max(relative_error)),
        "mismatched_rows": int(np.count_nonzero(~close)),
        "max_score_error": float(np.max(np.abs(expected_scores - actual_scores))),
        "threshold": threshold,
        "threshold_decision_mismatches": int(decision_mismatches),
        "worst_dataset_index": int(validation_indices[worst_index]),
        "worst_jax_logit": float(expected[worst_index]),
        "worst_onnx_logit": float(actual[worst_index]),
    }
    print(json.dumps(summary, indent=2))
    if not np.all(close):
        raise AssertionError(
            f"ONNX parity failed for {summary['mismatched_rows']} of "
            f"{summary['rows_compared']} rows."
        )
    if decision_mismatches:
        raise AssertionError(
            "ONNX parity changed "
            f"{int(decision_mismatches)} threshold decisions at {threshold:.6f}."
        )
    print("ONNX structure, checksums, and JAX logit parity passed.")


if __name__ == "__main__":
    main()
