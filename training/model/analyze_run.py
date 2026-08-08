"""Export per-window predictions and threshold analysis for a saved model run.

Run from the repository root:
    python training/model/analyze_run.py --run-dir models/runs/stableronaldo
"""

import argparse
import csv
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from architecture import ChatClassifier
from data import (
    iterate_batches,
    load_dataset_rows,
    load_vod_manifest,
    prepare_dataset_from_saved_preprocessing,
    streamer_holdout_split,
    vod_group_split,
    vod_manifest_split,
)
from evaluate import (
    confusion_counts,
    evaluate,
    find_best_threshold,
    precision_recall_f1,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validation_indices(rows, metadata, config):
    """Reconstruct the validation rows described by saved run metadata."""
    if metadata.get("holdout_vod_ids"):
        _, val_idx = vod_manifest_split(
            rows,
            metadata["holdout_vod_ids"],
        )
        return val_idx
    split = metadata["split"]
    if split.startswith("holdout streamer ="):
        streamer = split.partition("=")[2].strip()
        _, val_idx = streamer_holdout_split(rows, streamer)
        return val_idx
    if split.startswith("VOD-grouped"):
        val_fraction = float(config["training"].get("validation_fraction", 0.2))
        _, val_idx = vod_group_split(
            rows,
            val_frac=val_fraction,
            seed=int(metadata.get("seed", 0)),
        )
        return val_idx
    raise ValueError(
        f"Unsupported split {split!r}. Analyze a VOD-grouped or streamer-holdout run."
    )


def load_model(run_dir, metadata):
    """Recreate the architecture and deserialize its trained variable tree."""
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
    with open(
        os.path.join(run_dir, "chat_classifier_params.msgpack"),
        "rb",
    ) as f:
        variables = serialization.from_bytes(template, f.read())
    return model, variables


def predict(model, variables, tokens, features, labels, indices, batch_size):
    logits = []
    for batch_tokens, batch_features, _ in iterate_batches(
        tokens,
        features,
        labels,
        indices,
        batch_size,
        shuffle=False,
    ):
        batch_logits = model.apply(
            variables,
            jnp.asarray(batch_tokens),
            jnp.asarray(batch_features),
            training=False,
        )
        logits.append(np.asarray(batch_logits))
    logits = np.concatenate(logits)
    probabilities = np.asarray(jax.nn.sigmoid(jnp.asarray(logits)))
    return probabilities


def twitch_vod_url(vod_id, offset_seconds):
    seconds = max(0, int(offset_seconds or 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"https://www.twitch.tv/videos/{vod_id}?t={hours}h{minutes}m{seconds}s"


def outcome(label, predicted):
    if label == 1 and predicted == 1:
        return "TP"
    if label == 0 and predicted == 1:
        return "FP"
    if label == 1 and predicted == 0:
        return "FN"
    return "TN"


def build_prediction_rows(rows, val_idx, probabilities, threshold):
    predictions = []
    for dataset_index, probability in zip(val_idx, probabilities):
        row = rows[int(dataset_index)]
        label = int(row.get("label", 0))
        predicted = int(float(probability) >= threshold)
        vod_id = str(row.get("vod_id", "unknown"))
        target_offset = row.get("target_offset", 0)
        predictions.append(
            {
                "dataset_index": int(dataset_index),
                "outcome": outcome(label, predicted),
                "label": label,
                "predicted_label": predicted,
                "score": float(probability),
                "threshold": threshold,
                "streamer_name": row.get("streamer_name", "unknown"),
                "vod_id": vod_id,
                "target_offset": target_offset,
                "window_start": row.get("window_start"),
                "window_end": row.get("window_end"),
                "message_count": row.get("message_count"),
                "messages_per_second": row.get("messages_per_second"),
                "unique_users": row.get("unique_users"),
                "twitch_url": twitch_vod_url(vod_id, target_offset),
                "messages": row.get("messages", []),
            }
        )
    return sorted(predictions, key=lambda item: item["score"], reverse=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def threshold_metrics(labels, probabilities, threshold):
    precision, recall, f1 = precision_recall_f1(
        labels,
        probabilities,
        threshold=threshold,
    )
    tp, fp, fn, tn = confusion_counts(
        labels,
        probabilities,
        threshold=threshold,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def write_threshold_report(path, labels, probabilities):
    fieldnames = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for threshold in np.linspace(0.05, 0.95, 181):
            metrics = threshold_metrics(
                labels,
                probabilities,
                threshold=float(threshold),
            )
            writer.writerow(
                {
                    "threshold": float(threshold),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    **metrics["confusion"],
                }
            )


def precision_targets(labels, probabilities, targets=(0.5, 0.6, 0.7, 0.8)):
    """Find the highest-recall observed threshold meeting each precision target."""
    results = {}
    thresholds = np.unique(probabilities)
    for target in targets:
        best = None
        for threshold in thresholds:
            metrics = threshold_metrics(
                labels,
                probabilities,
                threshold=float(threshold),
            )
            predicted_positives = (
                metrics["confusion"]["tp"] + metrics["confusion"]["fp"]
            )
            if predicted_positives == 0 or metrics["precision"] < target:
                continue
            if best is None or (
                metrics["recall"],
                metrics["precision"],
            ) > (
                best["recall"],
                best["precision"],
            ):
                best = {
                    "threshold": float(threshold),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "confusion": metrics["confusion"],
                }
        results[f"{int(target * 100)}%"] = best
    return results


def write_review_template(path, false_positives, limit):
    fieldnames = [
        "review_label",
        "review_notes",
        "dataset_index",
        "streamer_name",
        "vod_id",
        "target_offset",
        "score",
        "twitch_url",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in false_positives[:limit]:
            writer.writerow(
                {
                    "review_label": "",
                    "review_notes": "",
                    "dataset_index": row["dataset_index"],
                    "streamer_name": row["streamer_name"],
                    "vod_id": row["vod_id"],
                    "target_offset": row["target_offset"],
                    "score": row["score"],
                    "twitch_url": row["twitch_url"],
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--vod-manifest",
        default=None,
        help=(
            "Evaluate this already-trained run on an external VOD manifest at "
            "its saved threshold."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <run-dir>/analysis",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the threshold saved with the run.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--review-limit", type=int, default=100)
    parser.add_argument(
        "--overwrite-review",
        action="store_true",
        help="Replace an existing false_positive_review.csv.",
    )
    parser.add_argument(
        "--explore-thresholds",
        action="store_true",
        help=(
            "Generate threshold tuning reports for an external VOD manifest. "
            "Doing this consumes it as a tuning set rather than an untouched test."
        ),
    )
    args = parser.parse_args()

    metadata = load_json(os.path.join(args.run_dir, "inference_meta.json"))
    if (
        int(metadata.get("window_seconds", 0)) != 35
        or int(metadata.get("target_lag_seconds", 0)) != 30
        or metadata.get("window_geometry") != "clip_start_minus_5_plus_30"
        or int(metadata.get("window_geometry_version", 0)) != 2
    ):
        raise ValueError(
            "saved run uses legacy or unknown window geometry; analyze its "
            "existing saved reports or retrain on the window-v2 dataset"
        )
    vocab = load_json(os.path.join(args.run_dir, "vocab.json"))
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rows = load_dataset_rows()
    external_test = args.vod_manifest is not None
    if external_test:
        external_vod_ids = load_vod_manifest(args.vod_manifest)
        _, val_idx = vod_manifest_split(rows, external_vod_ids)
        reviewed_holdout_rows = [
            int(index)
            for index in val_idx
            if rows[int(index)].get("review_label")
        ]
        if reviewed_holdout_rows:
            raise ValueError(
                f"{len(reviewed_holdout_rows)} external-test rows already have "
                "manual review labels. This manifest is not untouched."
            )
        evaluation_split = (
            f"external untouched VOD manifest = "
            f"{os.path.normpath(args.vod_manifest)}"
        )
    else:
        val_idx = validation_indices(rows, metadata, config)
        saved_confusion = metadata.get("best_val_metrics", {}).get("confusion", {})
        expected_validation_size = sum(
            int(saved_confusion.get(name, 0)) for name in ("tp", "fp", "fn", "tn")
        )
        if expected_validation_size and len(val_idx) != expected_validation_size:
            raise ValueError(
                "The processed dataset changed after this model was trained: "
                f"saved validation size={expected_validation_size}, "
                f"current reconstructed size={len(val_idx)}. Analyze the run against "
                "its original dataset or retrain it."
            )
        evaluation_split = metadata["split"]
    tokens, features, labels = prepare_dataset_from_saved_preprocessing(
        rows=rows,
        vocab=vocab,
        max_seq_len=int(metadata["max_seq_len"]),
        stream_time_scale_seconds=int(metadata["stream_time_scale_seconds"]),
        feature_mean=metadata["feature_mean"],
        feature_std=metadata["feature_std"],
    )
    model, variables = load_model(args.run_dir, metadata)
    probabilities = predict(
        model,
        variables,
        tokens,
        features,
        labels,
        val_idx,
        args.batch_size,
    )
    val_labels = labels[val_idx].astype(np.float32)
    if len(np.unique(val_labels)) < 2:
        raise ValueError(
            "The selected VODs do not contain both positive and negative examples; "
            "AUC/AP evaluation would not be meaningful."
        )

    if args.threshold is None:
        threshold = float(metadata["threshold"])
    else:
        threshold = args.threshold
    metrics = evaluate(val_labels, probabilities, threshold=threshold)
    explore_thresholds = not external_test or args.explore_thresholds
    if explore_thresholds:
        best_f1_threshold, best_f1_metrics = find_best_threshold(
            val_labels,
            probabilities,
        )
        target_metrics = precision_targets(val_labels, probabilities)
    else:
        best_f1_threshold = None
        best_f1_metrics = None
        target_metrics = None
    prediction_rows = build_prediction_rows(
        rows,
        val_idx,
        probabilities,
        threshold,
    )
    false_positives = [
        row for row in prediction_rows if row["outcome"] == "FP"
    ]
    false_negatives = [
        row for row in prediction_rows if row["outcome"] == "FN"
    ]

    if args.output_dir:
        output_dir = args.output_dir
    elif external_test:
        manifest_name = os.path.splitext(
            os.path.basename(args.vod_manifest)
        )[0]
        output_dir = os.path.join(args.run_dir, f"analysis-{manifest_name}")
    else:
        output_dir = os.path.join(args.run_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(
        os.path.join(output_dir, "validation_predictions.jsonl"),
        prediction_rows,
    )
    write_jsonl(
        os.path.join(output_dir, "false_positives.jsonl"),
        false_positives,
    )
    write_jsonl(
        os.path.join(output_dir, "false_negatives.jsonl"),
        false_negatives,
    )
    if explore_thresholds:
        write_threshold_report(
            os.path.join(output_dir, "threshold_report.csv"),
            val_labels,
            probabilities,
        )
    review_path = os.path.join(output_dir, "false_positive_review.csv")
    if args.overwrite_review or not os.path.exists(review_path):
        write_review_template(
            review_path,
            false_positives,
            args.review_limit,
        )
    else:
        print(f"Preserved existing review labels in {review_path}")

    summary = {
        "run_dir": args.run_dir,
        "training_split": metadata["split"],
        "evaluation_split": evaluation_split,
        "seed": metadata.get("seed"),
        "validation_examples": len(val_idx),
        "positive_prevalence": float(np.mean(val_labels)),
        "threshold": threshold,
        "metrics": metrics,
        "threshold_exploration_performed": explore_thresholds,
        "best_f1_threshold": best_f1_threshold,
        "best_f1_metrics": best_f1_metrics,
        "precision_targets": target_metrics,
        "review_instructions": {
            "positive": "Actually clip-worthy; current label is likely wrong.",
            "hard_negative": "Correctly negative despite a high model score.",
            "uncertain": "Cannot confidently decide from VOD/chat context.",
        },
    }
    with open(
        os.path.join(output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Analyzed {len(val_idx)} validation windows at threshold {threshold:.3f}")
    print(
        f"Precision={metrics['precision']:.3f} "
        f"Recall={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} "
        f"AUC={metrics['auc']:.3f} "
        f"AP={metrics['average_precision']:.3f}"
    )
    print(
        f"False positives: {len(false_positives)} | "
        f"False negatives: {len(false_negatives)}"
    )
    print(f"Analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
