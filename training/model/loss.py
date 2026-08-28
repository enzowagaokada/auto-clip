"""Numerically stable weighted binary cross-entropy on model logits."""

import jax.numpy as jnp
import optax


def weighted_bce(logits, labels, pos_weight, sample_weight=1.0):
    """Weighted BCE normalized by the effective batch weight.

    ``pos_weight`` is normally calculated from the training split rather than
    hardcoded. ``sample_weight`` is a per-example multiplier (for example the
    extra hard-negative weight). Keeping the model output as logits avoids
    saturated sigmoid/log gradients.
    """
    losses = optax.sigmoid_binary_cross_entropy(logits, labels)
    weights = (1.0 + labels * (pos_weight - 1.0)) * sample_weight
    return jnp.sum(losses * weights) / jnp.maximum(jnp.sum(weights), 1e-8)
