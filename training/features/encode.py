"""Encode the processed dataset into model-ready arrays.

Reads data/processed/dataset.jsonl and produces:
  - data/processed/vocab.json     token -> id mapping
  - data/processed/encoded.npz    tokens (int32), features (float32), labels (int32)
  - data/processed/meta.json      shapes, feature scaling, streamer per row

Run from the repository root: python training/features/encode.py
"""

import json
import os

import numpy as np
import yaml

from tokenizer import build_vocab, encode_window, save_vocab


DATASET_FILE = "data/processed/dataset.jsonl"
PROCESSED_DIR = "data/processed"
VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab.json")
ENCODED_FILE = os.path.join(PROCESSED_DIR, "encoded.npz")
META_FILE = os.path.join(PROCESSED_DIR, "meta.json")

MIN_FREQ = 2
MAX_VOCAB = 20000


def load_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found. Run build_dataset.py first.")
        return

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    max_seq_len = config["model"]["max_seq_len"]

    print(f"Loading {DATASET_FILE}...")
    rows = load_dataset(DATASET_FILE)
    print(f"Loaded {len(rows)} examples.")

    # Build vocabulary from the full chat corpus.
    print("Building vocabulary...")
    all_messages = (msg for row in rows for msg in row.get("messages", []))
    vocab = build_vocab(all_messages, min_freq=MIN_FREQ, max_size=MAX_VOCAB)
    print(f"Vocab size: {len(vocab)}")

    # Normalized stream time uses the max target offset across the dataset.
    max_offset = max((row.get("target_offset") or 0) for row in rows) or 1

    tokens = np.zeros((len(rows), max_seq_len), dtype=np.int32)
    raw_features = np.zeros((len(rows), 3), dtype=np.float32)
    labels = np.zeros((len(rows),), dtype=np.int32)
    streamers = []

    print("Encoding windows...")
    for i, row in enumerate(rows):
        tokens[i] = encode_window(row.get("messages", []), vocab, max_seq_len)

        mps = float(row.get("messages_per_second", 0.0))
        unique_users = float(row.get("unique_users", 0))
        stream_time = float(row.get("target_offset") or 0) / max_offset

        raw_features[i] = [mps, unique_users, stream_time]
        labels[i] = int(row.get("label", 0))
        streamers.append(row.get("streamer_name", "unknown"))

    # Standardize the two count-based features (stream_time is already 0-1).
    feat_mean = raw_features.mean(axis=0)
    feat_std = raw_features.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    # Leave normalized stream time (index 2) untouched.
    feat_mean[2] = 0.0
    feat_std[2] = 1.0

    features = (raw_features - feat_mean) / feat_std

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_vocab(vocab, VOCAB_FILE)
    np.savez(
        ENCODED_FILE,
        tokens=tokens,
        features=features.astype(np.float32),
        labels=labels,
    )

    meta = {
        "num_examples": len(rows),
        "max_seq_len": max_seq_len,
        "vocab_size": len(vocab),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "feature_names": ["messages_per_second", "unique_users", "normalized_stream_time"],
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "max_offset": int(max_offset),
        "streamers": streamers,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n--- Summary ---")
    print(f"tokens:   {tokens.shape} (int32)")
    print(f"features: {features.shape} (float32)")
    print(f"labels:   {labels.shape} -> {int(labels.sum())} pos / {int((labels == 0).sum())} neg")
    print(f"Saved: {VOCAB_FILE}, {ENCODED_FILE}, {META_FILE}")


if __name__ == "__main__":
    main()
