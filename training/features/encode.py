"""Encode the processed dataset into model-ready inspection arrays.

Reads data/processed/dataset.jsonl and produces:
  - data/processed/vocab.json     token -> id mapping
  - data/processed/encoded.npz    tokens (int32), features (float32), labels (int32)
  - data/processed/meta.json      shapes, feature scaling, streamer per row

Run from the repository root: python training/features/encode.py

The training script performs its own split-aware encoding so vocabulary and
feature statistics are fitted on training rows only. These full-dataset arrays
remain useful for inspection and backwards-compatible tooling.
"""

import json
import os

import numpy as np
import yaml

from preprocessing import FEATURE_NAMES, encode_rows, fit_feature_scaler, scale_features
from tokenizer import build_vocab, save_vocab


DATASET_FILE = "data/processed/dataset.jsonl"
PROCESSED_DIR = "data/processed"
VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab.json")
ENCODED_FILE = os.path.join(PROCESSED_DIR, "encoded.npz")
META_FILE = os.path.join(PROCESSED_DIR, "meta.json")

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
    model_config = config["model"]
    train_config = config["training"]
    max_seq_len = model_config["max_seq_len"]
    min_freq = int(model_config.get("min_token_frequency", 3))
    max_vocab = int(model_config.get("max_vocab_size", 10000))
    stream_time_scale = int(train_config.get("stream_time_scale_seconds", 43200))

    print(f"Loading {DATASET_FILE}...")
    rows = load_dataset(DATASET_FILE)
    print(f"Loaded {len(rows)} examples.")

    # Build vocabulary from the full chat corpus.
    print("Building vocabulary...")
    all_messages = (msg for row in rows for msg in row.get("messages", []))
    vocab = build_vocab(all_messages, min_freq=min_freq, max_size=max_vocab)
    print(f"Vocab size: {len(vocab)}")

    print("Encoding windows...")
    tokens, raw_features, labels = encode_rows(
        rows, vocab, max_seq_len, stream_time_scale
    )
    all_indices = np.arange(len(rows))
    feat_mean, feat_std = fit_feature_scaler(raw_features, all_indices)
    features = scale_features(raw_features, feat_mean, feat_std)

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
        "feature_names": FEATURE_NAMES,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "stream_time_scale_seconds": stream_time_scale,
        "streamers": [row.get("streamer_name", "unknown") for row in rows],
        "vod_ids": [str(row.get("vod_id", "unknown")) for row in rows],
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
