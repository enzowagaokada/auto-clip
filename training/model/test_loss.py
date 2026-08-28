import unittest

import jax.numpy as jnp
import numpy as np

from loss import weighted_bce


class WeightedLossTest(unittest.TestCase):
    def test_global_weight_scale_does_not_change_loss(self):
        logits = jnp.asarray([0.0, 1.0, -1.0])
        labels = jnp.asarray([0.0, 1.0, 0.0])
        weights = jnp.asarray([0.5, 1.0, 2.0])
        first = float(weighted_bce(logits, labels, 2.0, weights))
        second = float(weighted_bce(logits, labels, 2.0, weights * 10.0))
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
