"""Train the baseline GRU chat classifier with JAX/Flax/Optax.

Run from the repository root:
    python training/model/train.py
    python training/model/train.py --holdout-streamer jasontheween

Random split is the default baseline. The streamer-holdout option trains on all
other streamers and validates on the held-out one (the real generalization test).
"""

import argparse
import json
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
    iterate_batches,
    load_encoded,
    random_split,
    streamer_holdout_split,
)
from evaluate import evaluate, format_metrics
from loss import weighted_bce

MODELS_DIR = "models"


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_state(model, rng, max_seq_len, num_features, learning_rate):
    dummy_tokens = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
    dummy_features = jnp.zeros((1, num_features), dtype=jnp.float32)
    params = model.init(
        {"params": rng, "dropout": rng}, dummy_tokens, dummy_features, training=False
    )
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, tokens, features, labels, pos_weight, dropout_key):
    def loss_fn(params):
        preds = state.apply_fn(
            params, tokens, features, training=True, rngs={"dropout": dropout_key}
        )
        return weighted_bce(preds, labels, pos_weight)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def predict_batch(state, tokens, features):
    return state.apply_fn(state.params, tokens, features, training=False)


def run_predictions(state, tokens, features, labels, indices, batch_size):
    preds = []
    for bt, bf, _ in iterate_batches(
        tokens, features, labels, indices, batch_size, shuffle=False
    ):
        preds.append(np.asarray(predict_batch(state, jnp.asarray(bt), jnp.asarray(bf))))
    return np.concatenate(preds), labels[indices].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-streamer", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]

    tokens, features, labels, meta = load_encoded()
    num_examples = len(labels)
    num_features = features.shape[1]

    if args.holdout_streamer:
        train_idx, val_idx = streamer_holdout_split(meta, args.holdout_streamer)
        split_desc = f"holdout streamer = {args.holdout_streamer}"
    else:
        train_idx, val_idx = random_split(num_examples, val_frac=0.2, seed=args.seed)
        split_desc = "random 80/20 split"

    print(f"Examples: {num_examples} | {split_desc}")
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    pos_weight = float(train_cfg.get("pos_weight", 4.0))
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 20))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))

    model = ChatClassifier(
        vocab_size=meta["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_features=num_features,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = create_state(model, init_rng, model_cfg["max_seq_len"], num_features, learning_rate)

    best_f1 = -1.0
    best_params = None

    for epoch in range(epochs):
        rng, shuffle_rng = jax.random.split(rng)
        np_rng = np.random.default_rng(int(shuffle_rng[0]))

        epoch_losses = []
        for bt, bf, bl in iterate_batches(
            tokens, features, labels, train_idx, batch_size, rng=np_rng, shuffle=True
        ):
            rng, dropout_key = jax.random.split(rng)
            state, loss = train_step(
                state,
                jnp.asarray(bt),
                jnp.asarray(bf),
                jnp.asarray(bl),
                pos_weight,
                dropout_key,
            )
            epoch_losses.append(float(loss))

        val_preds, val_labels = run_predictions(
            state, tokens, features, labels, val_idx, batch_size
        )
        metrics = evaluate(val_labels, val_preds, threshold=args.threshold)
        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"Epoch {epoch + 1:2d} | loss {mean_loss:.4f} | val {format_metrics(metrics)}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_params = jax.device_get(state.params)

    os.makedirs(MODELS_DIR, exist_ok=True)
    params_path = os.path.join(MODELS_DIR, "chat_classifier_params.msgpack")
    with open(params_path, "wb") as f:
        f.write(serialization.to_bytes(best_params if best_params is not None else state.params))

    inference_meta = {
        "vocab_size": meta["vocab_size"],
        "embed_dim": model_cfg["embed_dim"],
        "hidden_dim": model_cfg["hidden_dim"],
        "max_seq_len": model_cfg["max_seq_len"],
        "num_features": num_features,
        "feature_names": meta["feature_names"],
        "feature_mean": meta["feature_mean"],
        "feature_std": meta["feature_std"],
        "max_offset": meta["max_offset"],
        "threshold": args.threshold,
        "best_val_f1": best_f1,
        "split": split_desc,
    }
    with open(os.path.join(MODELS_DIR, "inference_meta.json"), "w", encoding="utf-8") as f:
        json.dump(inference_meta, f, ensure_ascii=False, indent=2)

    print(f"\nBest val F1: {best_f1:.3f}")
    print(f"Saved params -> {params_path}")


if __name__ == "__main__":
    main()
