"""Weighted binary cross-entropy.

Positives (clippable moments) are rare, so we weight them more heavily to
penalize missed viral moments more than false alarms. Set pos_weight roughly to
the negative:positive ratio (~3.66 for the current dataset).
"""

import jax.numpy as jnp

EPS = 1e-7


def weighted_bce(preds, labels, pos_weight):
    """Mean weighted BCE over a batch.

    preds, labels: (batch,) float arrays. preds are probabilities in (0, 1).
    """
    preds = jnp.clip(preds, EPS, 1.0 - EPS)
    loss = -(
        pos_weight * labels * jnp.log(preds)
        + (1.0 - labels) * jnp.log(1.0 - preds)
    )
    return jnp.mean(loss)
