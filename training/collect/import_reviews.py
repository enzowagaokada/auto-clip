"""Import reviewed false positives into durable dataset annotations.

Run from the repository root:
    python training/collect/import_reviews.py \
        --review-file models/runs/stableronaldo/analysis/false_positive_review.csv
"""

import argparse
import csv
import json
import os
from collections import Counter


DEFAULT_OUTPUT = "data/reviews/window_labels.csv"
OUTPUT_FIELDS = [
    "streamer_name",
    "vod_id",
    "target_offset",
    "review_label",
    "training_label",
    "review_notes",
    "source_run",
    "dataset_index",
    "score",
    "twitch_url",
]
LABEL_ALIASES = {
    "positive": "positive",
    "hard_negative": "hard_negative",
    "hardnegative": "hard_negative",
    "uncertain": "uncertain",
    "unertain": "uncertain",
}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def flatten_values(row):
    values = []
    for value in row.values():
        if isinstance(value, list):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return values


def prediction_indexes(predictions):
    by_dataset_index = {
        int(row["dataset_index"]): row for row in predictions
    }
    by_url = {row["twitch_url"]: row for row in predictions}
    return by_dataset_index, by_url


def resolve_prediction(review_row, by_dataset_index, by_url):
    """Resolve even a mildly malformed edited CSV row against source predictions."""
    try:
        dataset_index = int(review_row.get("dataset_index", ""))
    except (TypeError, ValueError):
        dataset_index = None
    if dataset_index in by_dataset_index:
        return by_dataset_index[dataset_index]

    for value in flatten_values(review_row):
        value = str(value).strip()
        if value in by_url:
            return by_url[value]

    raise ValueError(
        "Could not match reviewed row to its source prediction: "
        f"{review_row!r}"
    )


def normalize_label(value):
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    if normalized not in LABEL_ALIASES:
        raise ValueError(
            f"Unknown review label {value!r}; expected positive, "
            "hard_negative, or uncertain."
        )
    return LABEL_ALIASES[normalized]


def annotation_key(row):
    return (
        str(row["streamer_name"]).strip().lower(),
        str(row["vod_id"]),
        int(float(row["target_offset"])),
    )


def load_existing_annotations(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {
            annotation_key(row): row
            for row in csv.DictReader(f)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-file", required=True)
    parser.add_argument(
        "--predictions-file",
        default=None,
        help="Default: false_positives.jsonl beside the review CSV.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predictions_file = args.predictions_file or os.path.join(
        os.path.dirname(args.review_file),
        "false_positives.jsonl",
    )
    predictions = load_jsonl(predictions_file)
    by_dataset_index, by_url = prediction_indexes(predictions)
    annotations = load_existing_annotations(args.output)
    source_run = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.abspath(args.review_file)))
    )

    imported = Counter()
    with open(args.review_file, "r", encoding="utf-8", newline="") as f:
        for review_row in csv.DictReader(f):
            review_label = normalize_label(review_row.get("review_label"))
            if review_label is None:
                continue

            prediction = resolve_prediction(
                review_row,
                by_dataset_index,
                by_url,
            )
            training_label = {
                "positive": "1",
                "hard_negative": "0",
                "uncertain": "",
            }[review_label]
            annotation = {
                "streamer_name": prediction["streamer_name"],
                "vod_id": str(prediction["vod_id"]),
                "target_offset": int(prediction["target_offset"]),
                "review_label": review_label,
                "training_label": training_label,
                "review_notes": str(review_row.get("review_notes") or "").strip(),
                "source_run": source_run,
                "dataset_index": int(prediction["dataset_index"]),
                "score": float(prediction["score"]),
                "twitch_url": prediction["twitch_url"],
            }
            annotations[annotation_key(annotation)] = annotation
            imported[review_label] += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    temporary_output = args.output + ".tmp"
    with open(temporary_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for annotation in sorted(annotations.values(), key=annotation_key):
            writer.writerow(annotation)
    os.replace(temporary_output, args.output)

    print(
        f"Imported {sum(imported.values())} reviews: "
        f"positive={imported['positive']} "
        f"hard_negative={imported['hard_negative']} "
        f"uncertain={imported['uncertain']}"
    )
    print(f"Durable annotations: {len(annotations)} -> {args.output}")


if __name__ == "__main__":
    main()
