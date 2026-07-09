"""Load encoded arrays and produce train/val splits and minibatches."""

import json
import os

import numpy as np

PROCESSED_DIR = "data/processed"
ENCODED_FILE = os.path.join(PROCESSED_DIR, "encoded.npz")
META_FILE = os.path.join(PROCESSED_DIR, "meta.json")


def load_encoded():
    """Return (tokens, features, labels, meta)."""
    data = np.load(ENCODED_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return data["tokens"], data["features"], data["labels"], meta


def random_split(num_examples, val_frac=0.2, seed=0):
    """Shuffled index split. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_examples)
    n_val = int(round(num_examples * val_frac))
    return idx[n_val:], idx[:n_val]


def streamer_holdout_split(meta, holdout_streamer):
    """Split so one streamer is entirely in validation (generalization test)."""
    streamers = np.array(meta["streamers"])
    val_mask = streamers == holdout_streamer
    all_idx = np.arange(len(streamers))
    return all_idx[~val_mask], all_idx[val_mask]


def iterate_batches(tokens, features, labels, indices, batch_size, rng=None, shuffle=True):
    """Yield (tokens, features, labels) minibatches for the given indices."""
    indices = np.array(indices)
    if shuffle:
        rng = rng or np.random.default_rng(0)
        indices = rng.permutation(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield (
            tokens[batch_idx],
            features[batch_idx],
            labels[batch_idx].astype(np.float32),
        )
