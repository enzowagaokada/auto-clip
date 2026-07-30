"""Numerically stable weighted binary cross-entropy on model logits."""

import jax.numpy as jnp
import optax


def weighted_bce(logits, labels, pos_weight):
    """Mean weighted BCE over a batch.

    ``pos_weight`` is normally calculated from the training split rather than
    hardcoded. Keeping the model output as logits avoids saturated sigmoid/log
    gradients.
    """
    losses = optax.sigmoid_binary_cross_entropy(logits, labels)
    weights = 1.0 + labels * (pos_weight - 1.0)
    return jnp.mean(losses * weights)
