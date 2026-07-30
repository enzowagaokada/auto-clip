"""Load rows, create leakage-resistant splits, and prepare minibatches."""

import json
import os
import sys
from collections import defaultdict

import numpy as np

PROCESSED_DIR = "data/processed"
ENCODED_FILE = os.path.join(PROCESSED_DIR, "encoded.npz")
META_FILE = os.path.join(PROCESSED_DIR, "meta.json")
DATASET_FILE = os.path.join(PROCESSED_DIR, "dataset.jsonl")

FEATURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "features"))
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)

from preprocessing import FEATURE_NAMES, encode_rows, fit_feature_scaler, scale_features
from tokenizer import build_vocab


def load_encoded():
    """Return (tokens, features, labels, meta)."""
    data = np.load(ENCODED_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return data["tokens"], data["features"], data["labels"], meta


def load_dataset_rows(path=DATASET_FILE):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_vod_manifest(path):
    """Load one VOD ID per line, allowing blank lines and # comments."""
    vod_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.partition("#")[0].strip()
            if value:
                vod_ids.add(value)
    if not vod_ids:
        raise ValueError(f"VOD manifest is empty: {path}")
    return vod_ids


def vod_manifest_split(rows, holdout_vod_ids):
    """Return train/holdout indices for an explicit set of complete VODs."""
    requested = {str(vod_id) for vod_id in holdout_vod_ids}
    present = {str(row.get("vod_id", "unknown")) for row in rows}
    missing = sorted(requested - present)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} manifest VODs are absent from the processed dataset: "
            f"{preview}{suffix}"
        )

    val_mask = np.array(
        [str(row.get("vod_id", "unknown")) in requested for row in rows],
        dtype=bool,
    )
    if not np.any(val_mask):
        raise ValueError("The VOD manifest selected no validation examples.")
    if np.all(val_mask):
        raise ValueError("The VOD manifest selected every example; training would be empty.")
    all_idx = np.arange(len(rows))
    return all_idx[~val_mask], all_idx[val_mask]


def streamer_holdout_split(rows, holdout_streamer):
    """Split so one streamer is entirely in validation (generalization test)."""
    streamers = np.array([row.get("streamer_name", "unknown") for row in rows])
    val_mask = streamers == holdout_streamer
    if not np.any(val_mask):
        available = ", ".join(sorted(set(streamers)))
        raise ValueError(
            f"No rows found for streamer {holdout_streamer!r}. Available: {available}"
        )
    all_idx = np.arange(len(streamers))
    return all_idx[~val_mask], all_idx[val_mask]


def vod_group_split(rows, val_frac=0.2, seed=0, attempts=256):
    """Split whole VODs while approximately preserving size and prevalence."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("vod_id", "unknown"))].append(index)

    vod_ids = np.array(list(groups))
    if len(vod_ids) < 2:
        raise ValueError("At least two distinct VODs are required for a VOD split.")

    labels = np.array([int(row.get("label", 0)) for row in rows])
    target_size = len(rows) * val_frac
    target_prevalence = float(labels.mean())
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(attempts):
        selected = []
        selected_size = 0
        for vod_id in rng.permutation(vod_ids):
            if selected and selected_size >= target_size:
                break
            selected.append(vod_id)
            selected_size += len(groups[vod_id])

        if len(selected) == len(vod_ids):
            selected = selected[:-1]
        val_idx = np.array(
            [index for vod_id in selected for index in groups[vod_id]], dtype=np.int32
        )
        if len(val_idx) == 0:
            continue

        val_prevalence = float(labels[val_idx].mean())
        size_error = abs(len(val_idx) - target_size) / len(rows)
        prevalence_error = abs(val_prevalence - target_prevalence)
        score = size_error + prevalence_error
        if best is None or score < best[0]:
            best = (score, val_idx)

    if best is None:
        raise ValueError("Could not construct a non-empty VOD validation split.")

    val_idx = best[1]
    val_mask = np.zeros(len(rows), dtype=bool)
    val_mask[val_idx] = True
    all_idx = np.arange(len(rows))
    return all_idx[~val_mask], all_idx[val_mask]


def prepare_dataset(
    rows,
    train_idx,
    max_seq_len,
    min_token_frequency,
    max_vocab_size,
    stream_time_scale_seconds,
):
    """Fit vocabulary/scaling on training rows and encode all rows."""
    train_messages = (
        message
        for index in train_idx
        for message in rows[int(index)].get("messages", [])
    )
    vocab = build_vocab(
        train_messages,
        min_freq=min_token_frequency,
        max_size=max_vocab_size,
    )
    tokens, raw_features, labels = encode_rows(
        rows,
        vocab,
        max_seq_len,
        stream_time_scale_seconds,
    )
    feature_mean, feature_std = fit_feature_scaler(raw_features, train_idx)
    features = scale_features(raw_features, feature_mean, feature_std)
    return tokens, features, labels, vocab, feature_mean, feature_std


def prepare_dataset_from_saved_preprocessing(
    rows,
    vocab,
    max_seq_len,
    stream_time_scale_seconds,
    feature_mean,
    feature_std,
):
    """Encode rows using the exact vocabulary/scaler shipped with a saved run."""
    tokens, raw_features, labels = encode_rows(
        rows,
        vocab,
        max_seq_len,
        stream_time_scale_seconds,
    )
    features = scale_features(
        raw_features,
        np.asarray(feature_mean, dtype=np.float32),
        np.asarray(feature_std, dtype=np.float32),
    )
    return tokens, features, labels


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
