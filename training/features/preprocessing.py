"""Shared feature extraction and split-aware array preparation."""

import numpy as np

from tokenizer import encode_window


TEMPORAL_BUCKET_COUNT = 7
FEATURE_NAMES = [
    "messages_per_second",
    "unique_users",
    "normalized_stream_time",
    *[f"message_rate_{i * 5}_{(i + 1) * 5}s" for i in range(TEMPORAL_BUCKET_COUNT)],
    "message_rate_change",
    "peak_5s_rate",
    "repeat_message_ratio",
]
STREAM_TIME_FEATURE_INDEX = 2


def raw_feature_vector(row, stream_time_scale_seconds):
    """Convert one processed row into the numeric model feature vector."""
    bucket_rates = list(row.get("message_rate_buckets", []))
    bucket_rates = (bucket_rates + [0.0] * TEMPORAL_BUCKET_COUNT)[:TEMPORAL_BUCKET_COUNT]
    stream_time = float(row.get("target_offset") or 0) / max(
        1.0, float(stream_time_scale_seconds)
    )
    stream_time = float(np.clip(stream_time, 0.0, 1.0))

    return [
        float(row.get("messages_per_second", 0.0)),
        float(row.get("unique_users", 0)),
        stream_time,
        *[float(value) for value in bucket_rates],
        float(row.get("message_rate_change", 0.0)),
        float(row.get("peak_5s_rate", 0.0)),
        float(row.get("repeat_message_ratio", 0.0)),
    ]


def encode_rows(rows, vocab, max_seq_len, stream_time_scale_seconds):
    """Encode text and unscaled numeric features for all rows."""
    tokens = np.zeros((len(rows), max_seq_len), dtype=np.int32)
    raw_features = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.zeros((len(rows),), dtype=np.int32)

    for index, row in enumerate(rows):
        tokens[index] = encode_window(row.get("messages", []), vocab, max_seq_len)
        raw_features[index] = raw_feature_vector(row, stream_time_scale_seconds)
        labels[index] = int(row.get("label", 0))

    return tokens, raw_features, labels


def fit_feature_scaler(raw_features, train_indices):
    """Fit means/stds using training rows only."""
    train_features = raw_features[np.asarray(train_indices)]
    means = train_features.mean(axis=0)
    stds = train_features.std(axis=0)
    stds[stds == 0] = 1.0

    # This feature is already bounded to [0, 1] using a fixed time scale.
    means[STREAM_TIME_FEATURE_INDEX] = 0.0
    stds[STREAM_TIME_FEATURE_INDEX] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def scale_features(raw_features, means, stds):
    return ((raw_features - means) / stds).astype(np.float32)
