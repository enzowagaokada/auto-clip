"""Train the split-aware GRU chat classifier with JAX/Flax/Optax.

Run from the repository root:
    python training/model/train.py
    python training/model/train.py --holdout-streamer jasontheween

Whole-VOD validation is the default. The streamer-holdout option trains on all
other streamers and validates on the held-out one.
"""

import argparse
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import serialization
from flax.training import train_state

from architecture import ChatClassifier
from data import (
    FEATURE_NAMES,
    iterate_batches,
    load_dataset_rows,
    load_vod_manifest,
    prepare_dataset,
    streamer_holdout_split,
    vod_group_split,
    vod_manifest_split,
)
from evaluate import evaluate, find_best_threshold, format_metrics
from loss import weighted_bce

MODELS_DIR = "models/runs/window-v2-vod-seed0"


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_state(
    model,
    rng,
    max_seq_len,
    num_features,
    learning_rate,
    weight_decay,
):
    dummy_tokens = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
    dummy_features = jnp.zeros((1, num_features), dtype=jnp.float32)
    params = model.init(
        {"params": rng, "dropout": rng}, dummy_tokens, dummy_features, training=False
    )
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, tokens, features, labels, pos_weight, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn(
            params, tokens, features, training=True, rngs={"dropout": dropout_key}
        )
        return weighted_bce(logits, labels, pos_weight)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def predict_logits_batch(state, tokens, features):
    return state.apply_fn(state.params, tokens, features, training=False)


def run_predictions(state, tokens, features, labels, indices, batch_size):
    logits = []
    for bt, bf, _ in iterate_batches(
        tokens, features, labels, indices, batch_size, shuffle=False
    ):
        logits.append(
            np.asarray(predict_logits_batch(state, jnp.asarray(bt), jnp.asarray(bf)))
        )
    logits = np.concatenate(logits)
    probabilities = np.asarray(jax.nn.sigmoid(jnp.asarray(logits)))
    return logits, probabilities, labels[indices].astype(np.float32)


def binary_cross_entropy(logits, labels):
    """Stable unweighted BCE for comparable train/validation diagnostics."""
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def main():
    parser = argparse.ArgumentParser()
    holdout_group = parser.add_mutually_exclusive_group()
    holdout_group.add_argument("--holdout-streamer", default=None)
    holdout_group.add_argument(
        "--holdout-vods",
        default=None,
        help=(
            "VOD manifest used as validation for early stopping/threshold tuning. "
            "For a truly untouched test, use analyze_run.py --vod-manifest instead."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=MODELS_DIR,
        help="Directory for params, deployable vocabulary, and inference metadata.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed evaluation threshold. Default: tune F1 on validation.",
    )
    args = parser.parse_args()

    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]
    window_seconds = int(train_cfg.get("window_seconds", 0))
    target_lag_seconds = int(train_cfg.get("target_lag_seconds", 0))
    if window_seconds != 35 or target_lag_seconds != 30:
        raise ValueError(
            "training window contract must be 35 seconds with a 30-second "
            "target lag ([clip start - 5s, clip start + 30s])"
        )
    rows = load_dataset_rows()
    for index, row in enumerate(rows):
        try:
            target = float(row["target_offset"])
            start = float(row["window_start"])
            end = float(row["window_end"])
            geometry_is_current = (
                row["window_geometry"] == "clip_start_minus_5_plus_30"
                and int(row["window_geometry_version"]) == 2
                and math.isclose(target - start, 5.0, abs_tol=1e-6)
                and math.isclose(end - target, 30.0, abs_tol=1e-6)
            )
        except (KeyError, TypeError, ValueError):
            geometry_is_current = False
        if not geometry_is_current:
            raise ValueError(
                f"dataset row {index} does not use window geometry v2; "
                "re-run build_dataset.py before training"
            )
    num_examples = len(rows)
    explicit_holdout_vod_ids = None

    if args.holdout_vods:
        explicit_holdout_vod_ids = load_vod_manifest(args.holdout_vods)
        train_idx, val_idx = vod_manifest_split(
            rows,
            explicit_holdout_vod_ids,
        )
        split_desc = f"holdout VOD manifest = {os.path.normpath(args.holdout_vods)}"
    elif args.holdout_streamer:
        train_idx, val_idx = streamer_holdout_split(rows, args.holdout_streamer)
        split_desc = f"holdout streamer = {args.holdout_streamer}"
    else:
        val_fraction = float(train_cfg.get("validation_fraction", 0.2))
        train_idx, val_idx = vod_group_split(
            rows,
            val_frac=val_fraction,
            seed=args.seed,
        )
        split_desc = f"VOD-grouped {1 - val_fraction:.0%}/{val_fraction:.0%} split"

    tokens, features, labels, vocab, feature_mean, feature_std = prepare_dataset(
        rows=rows,
        train_idx=train_idx,
        max_seq_len=int(model_cfg["max_seq_len"]),
        min_token_frequency=int(model_cfg.get("min_token_frequency", 3)),
        max_vocab_size=int(model_cfg.get("max_vocab_size", 10000)),
        stream_time_scale_seconds=int(
            train_cfg.get("stream_time_scale_seconds", 43200)
        ),
    )
    num_features = features.shape[1]

    print(f"Examples: {num_examples} | {split_desc}")
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    configured_pos_weight = train_cfg.get("pos_weight", "auto")
    if str(configured_pos_weight).lower() == "auto":
        train_labels = labels[train_idx]
        positives = int(np.sum(train_labels == 1))
        negatives = int(np.sum(train_labels == 0))
        if positives == 0:
            raise ValueError("Training split has no positive examples.")
        pos_weight = negatives / positives
    else:
        pos_weight = float(configured_pos_weight)

    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 20))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 3))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))
    selection_metric = str(train_cfg.get("selection_metric", "average_precision"))

    print(
        f"Vocab: {len(vocab)}  Features: {num_features}  "
        f"Positive weight: {pos_weight:.3f}"
    )

    model = ChatClassifier(
        vocab_size=len(vocab),
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_features=num_features,
        embedding_dropout_rate=float(
            model_cfg.get("embedding_dropout_rate", 0.15)
        ),
        head_dropout_rate=float(model_cfg.get("head_dropout_rate", 0.4)),
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_state(
        model,
        init_rng,
        model_cfg["max_seq_len"],
        num_features,
        learning_rate,
        weight_decay,
    )

    best_score = -float("inf")
    best_params = None
    best_metrics = None
    best_threshold = args.threshold if args.threshold is not None else 0.5
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        rng, shuffle_rng = jax.random.split(rng)
        np_rng = np.random.default_rng(int(shuffle_rng[0]))

        for bt, bf, bl in iterate_batches(
            tokens, features, labels, train_idx, batch_size, rng=np_rng, shuffle=True
        ):
            rng, dropout_key = jax.random.split(rng)
            state, _ = train_step(
                state,
                jnp.asarray(bt),
                jnp.asarray(bf),
                jnp.asarray(bl),
                pos_weight,
                dropout_key,
            )

        train_logits, train_preds, train_labels = run_predictions(
            state, tokens, features, labels, train_idx, batch_size
        )
        val_logits, val_preds, val_labels = run_predictions(
            state, tokens, features, labels, val_idx, batch_size
        )
        if args.threshold is None:
            threshold, metrics = find_best_threshold(val_labels, val_preds)
        else:
            threshold = args.threshold
            metrics = evaluate(val_labels, val_preds, threshold=threshold)
        train_metrics = evaluate(train_labels, train_preds, threshold=threshold)
        train_bce = binary_cross_entropy(train_logits, train_labels)
        val_bce = binary_cross_entropy(val_logits, val_labels)

        print(
            f"Epoch {epoch + 1:2d} | "
            f"train BCE={train_bce:.4f} {format_metrics(train_metrics)} | "
            f"val BCE={val_bce:.4f} {format_metrics(metrics)} | t={threshold:.3f}"
        )

        score = float(metrics.get(selection_metric, metrics["average_precision"]))
        if np.isnan(score):
            score = metrics["f1"]
        if score > best_score + min_delta:
            best_score = score
            best_params = jax.device_get(state.params)
            best_metrics = metrics
            best_threshold = threshold
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {patience} epochs without "
                    f"{selection_metric} improvement."
                )
                break

    os.makedirs(args.output_dir, exist_ok=True)
    params_path = os.path.join(args.output_dir, "chat_classifier_params.msgpack")
    with open(params_path, "wb") as f:
        f.write(
            serialization.to_bytes(
                best_params if best_params is not None else state.params
            )
        )
    with open(
        os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    inference_meta = {
        "vocab_size": len(vocab),
        "embed_dim": model_cfg["embed_dim"],
        "hidden_dim": model_cfg["hidden_dim"],
        "max_seq_len": model_cfg["max_seq_len"],
        "num_features": num_features,
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "window_seconds": window_seconds,
        "target_lag_seconds": target_lag_seconds,
        "window_geometry": "clip_start_minus_5_plus_30",
        "window_geometry_version": 2,
        "stream_time_scale_seconds": int(
            train_cfg.get("stream_time_scale_seconds", 43200)
        ),
        "threshold": best_threshold,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "selection_metric": selection_metric,
        "pos_weight": pos_weight,
        "split": split_desc,
        "holdout_vod_ids": (
            sorted(explicit_holdout_vod_ids)
            if explicit_holdout_vod_ids is not None
            else None
        ),
        "seed": args.seed,
    }
    with open(
        os.path.join(args.output_dir, "inference_meta.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(inference_meta, f, ensure_ascii=False, indent=2)

    print(
        f"\nBest epoch: {best_epoch} | threshold={best_threshold:.3f} | "
        f"{selection_metric}={best_score:.3f}"
    )
    print(f"Saved params -> {params_path}")


if __name__ == "__main__":
    main()
