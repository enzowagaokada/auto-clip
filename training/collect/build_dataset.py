import json
import os

import pandas as pd


CLIPS_FILE = "data/raw/clips.csv"
POSITIVE_DIR = "data/raw/chat"
NEGATIVE_DIR = "data/raw/chat_negatives"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.jsonl")


def load_clip_streamers():
    """Map clip_id -> streamer_name from clips.csv (positives lack this field)."""
    df = pd.read_csv(CLIPS_FILE)
    return dict(zip(df["clip_id"].astype(str), df["streamer_name"].astype(str)))


def compute_features(record):
    """Compute window-level features shared by positive and negative examples."""
    messages = record.get("messages", [])
    message_count = len(messages)

    window_start = record.get("window_start", 0)
    window_end = record.get("window_end", 0)
    duration = max(1, window_end - window_start)

    unique_users = len({m.get("user") for m in messages})

    return {
        "message_count": message_count,
        "messages_per_second": round(message_count / duration, 4),
        "unique_users": unique_users,
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    examples = []
    skipped_empty = 0
    missing_streamer = 0

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

    # Negatives (label = 0). These already carry label and streamer_name.
    for path in iter_json_files(NEGATIVE_DIR):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

        if not record.get("messages"):
            skipped_empty += 1
            continue

        streamer_name = record.get("streamer_name", "unknown")
        examples.append(build_example(record, label=0, streamer_name=streamer_name))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    positives = sum(1 for e in examples if e["label"] == 1)
    negatives = sum(1 for e in examples if e["label"] == 0)

    print("\n--- Summary ---")
    print(f"Total examples: {len(examples)}")
    print(f"Positives (label=1): {positives}")
    print(f"Negatives (label=0): {negatives}")
    if positives:
        print(f"Negative:positive ratio: {negatives / positives:.2f}:1")
    print(f"Skipped (no messages): {skipped_empty}")
    if missing_streamer:
        print(f"Positives with no streamer match in clips.csv: {missing_streamer}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
