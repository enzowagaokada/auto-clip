import csv
import json
import os

import pandas as pd


CLIPS_FILE = "data/raw/clips.csv"
POSITIVE_DIR = "data/raw/chat"
NEGATIVE_DIR = "data/raw/chat_negatives"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.jsonl")
REVIEW_LABELS_FILE = "data/reviews/window_labels.csv"
TEMPORAL_BUCKET_SECONDS = 5
TEMPORAL_BUCKET_COUNT = 7


def load_clip_streamers():
    """Map clip_id -> streamer_name from clips.csv (positives lack this field)."""
    df = pd.read_csv(CLIPS_FILE)
    return dict(zip(df["clip_id"].astype(str), df["streamer_name"].astype(str)))


def example_key(record):
    return (
        str(record.get("streamer_name", "unknown")).strip().lower(),
        str(record.get("vod_id")),
        int(float(record.get("target_offset") or 0)),
    )


def load_review_annotations():
    if not os.path.exists(REVIEW_LABELS_FILE):
        return {}
    with open(REVIEW_LABELS_FILE, "r", encoding="utf-8", newline="") as f:
        return {
            example_key(row): row
            for row in csv.DictReader(f)
        }


def apply_review_annotations(examples, annotations):
    """Override reviewed labels and exclude genuinely uncertain reviewed rows."""
    reviewed_examples = []
    counts = {"positive": 0, "hard_negative": 0, "uncertain": 0}
    for example in examples:
        annotation = annotations.get(example_key(example))
        if annotation is None:
            reviewed_examples.append(example)
            continue

        review_label = annotation["review_label"]
        counts[review_label] = counts.get(review_label, 0) + 1
        if review_label == "uncertain":
            continue

        example["label"] = int(annotation["training_label"])
        example["review_label"] = review_label
        example["review_notes"] = annotation.get("review_notes", "")
        reviewed_examples.append(example)
    return reviewed_examples, counts


def compute_features(record):
    """Compute window-level features shared by positive and negative examples."""
    messages = record.get("messages", [])
    message_count = len(messages)

    window_start = record.get("window_start", 0)
    window_end = record.get("window_end", 0)
    duration = max(1, window_end - window_start)

    unique_users = len({m.get("user") for m in messages if m.get("user")})
    bucket_counts = [0] * TEMPORAL_BUCKET_COUNT
    normalized_messages = []
    for message in messages:
        offset = float(message.get("offset_seconds", window_start))
        relative_offset = max(0.0, offset - window_start)
        bucket_index = min(
            int(relative_offset // TEMPORAL_BUCKET_SECONDS),
            TEMPORAL_BUCKET_COUNT - 1,
        )
        bucket_counts[bucket_index] += 1
        text = message.get("message", "").strip().casefold()
        if text:
            normalized_messages.append(text)

    bucket_rates = [
        round(count / TEMPORAL_BUCKET_SECONDS, 4) for count in bucket_counts
    ]
    early_rate = sum(bucket_rates[:2]) / 2
    recent_rate = sum(bucket_rates[-2:]) / 2
    repeat_ratio = (
        1.0 - len(set(normalized_messages)) / len(normalized_messages)
        if normalized_messages
        else 0.0
    )

    return {
        "message_count": message_count,
        "messages_per_second": round(message_count / duration, 4),
        "unique_users": unique_users,
        "message_rate_buckets": bucket_rates,
        "message_rate_change": round(recent_rate - early_rate, 4),
        "peak_5s_rate": max(bucket_rates, default=0.0),
        "repeat_message_ratio": round(repeat_ratio, 4),
        "window_start": window_start,
        "window_end": window_end,
    }


def build_example(record, label, streamer_name):
    """Assemble one dataset row from a raw chat window."""
    features = compute_features(record)

    return {
        "label": label,
        "streamer_name": streamer_name,
        "vod_id": str(record.get("vod_id")),
        "target_offset": record.get("target_offset"),
        "message_count": features["message_count"],
        "messages_per_second": features["messages_per_second"],
        "unique_users": features["unique_users"],
        "message_rate_buckets": features["message_rate_buckets"],
        "message_rate_change": features["message_rate_change"],
        "peak_5s_rate": features["peak_5s_rate"],
        "repeat_message_ratio": features["repeat_message_ratio"],
        "window_start": features["window_start"],
        "window_end": features["window_end"],
        "messages": [m.get("message", "") for m in record.get("messages", [])],
    }


def iter_json_files(directory):
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            yield os.path.join(directory, name)


def main():
    if not os.path.exists(CLIPS_FILE):
        print(f"Error: {CLIPS_FILE} not found. Run fetch_clips.py first.")
        return

    clip_streamers = load_clip_streamers()
    review_annotations = load_review_annotations()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    examples = []
    skipped_empty = 0
    missing_streamer = 0
    processed_files = 0

    # Positives (label = 1). These lack label/streamer_name, so we join on clip_id.
    for path in iter_json_files(POSITIVE_DIR):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        if not record.get("messages"):
            skipped_empty += 1
            continue

        clip_id = str(record.get("clip_id"))
        streamer_name = clip_streamers.get(clip_id)
        if streamer_name is None:
            missing_streamer += 1
            streamer_name = "unknown"

        examples.append(build_example(record, label=1, streamer_name=streamer_name))
        processed_files += 1
        if processed_files % 500 == 0:
            print(f"Processed {processed_files} non-empty chat files...")

    # Negatives (label = 0). These already carry label and streamer_name.
    for path in iter_json_files(NEGATIVE_DIR):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        if not record.get("messages"):
            skipped_empty += 1
            continue

        streamer_name = record.get("streamer_name", "unknown")
        examples.append(build_example(record, label=0, streamer_name=streamer_name))
        processed_files += 1
        if processed_files % 500 == 0:
            print(f"Processed {processed_files} non-empty chat files...")

    examples, review_counts = apply_review_annotations(
        examples,
        review_annotations,
    )

    temporary_output = OUTPUT_FILE + ".tmp"
    with open(temporary_output, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(temporary_output, OUTPUT_FILE)

    positives = sum(1 for e in examples if e["label"] == 1)
    negatives = sum(1 for e in examples if e["label"] == 0)

    print("\n--- Summary ---")
    print(f"Total examples: {len(examples)}")
    print(f"Positives (label=1): {positives}")
    print(f"Negatives (label=0): {negatives}")
    if positives:
        print(f"Negative:positive ratio: {negatives / positives:.2f}:1")
    print(f"Skipped (no messages): {skipped_empty}")
    if review_annotations:
        print(
            "Applied reviews: "
            f"{review_counts['positive']} positive, "
            f"{review_counts['hard_negative']} hard negative, "
            f"{review_counts['uncertain']} uncertain excluded"
        )
    if missing_streamer:
        print(f"Positives with no streamer match in clips.csv: {missing_streamer}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
