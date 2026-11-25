"""Checks for the grouping and aggregation rules in the seed replication."""

from unittest import TestCase

import numpy as np
import pandas as pd

from folding_seed_robustness import DEPTH_SPLIT, RANDOM_SPLIT, shared_splits, summarize_pairs


class FoldingSeedRobustnessTests(TestCase):
    def test_group_assignments_depend_only_on_the_pool(self):
        pool = pd.DataFrame({
            "circuit_id": np.repeat(np.arange(20), 2),
            "depth": np.repeat(np.tile([2, 3, 4, 5, 6], 4), 2),
        })
        splits = shared_splits(pool)
        expected = shared_splits(pool.assign(noisy_l1=np.linspace(-1, 1, len(pool))))

        for split, (tr, te, tags) in splits.items():
            np.testing.assert_array_equal(tr, expected[split][0])
            np.testing.assert_array_equal(te, expected[split][1])
            self.assertEqual(tags, expected[split][2])
            self.assertFalse(set(pool.iloc[tr].circuit_id) & set(pool.iloc[te].circuit_id))
        self.assertEqual(splits[RANDOM_SPLIT][2]["n_test_circuits"], 6)
        tr, te, _ = splits[DEPTH_SPLIT]
        self.assertLessEqual(pool.iloc[tr].depth.max(), 4)
        self.assertGreaterEqual(pool.iloc[te].depth.min(), 5)

    def test_overall_mean_uses_circuit_pools_as_units(self):
        pairs = pd.DataFrame({
            "split": [RANDOM_SPLIT] * 4,
            "comparison": ["folding_minus_parameter"] * 4,
            "model": ["rbf_krr"] * 4,
            "circuit_seed": [31, 31, 31, 32],
            "shot_seed": [11, 12, 13, 11],
            "diff": [1.0, 2.0, 3.0, -1.0],
            "diff_lo": [0.0, 0.2, 1.0, -2.0],
            "diff_hi": [2.0, 3.0, 4.0, -0.1],
        })
        summary, pool_summary = summarize_pairs(pairs)

        np.testing.assert_array_equal(pool_summary.diff_mean, [2.0, -1.0])
        self.assertEqual(summary.diff_mean.item(), 0.5)
        self.assertEqual(summary.n_resolved_positive.item(), 2)
        self.assertEqual(summary.n_resolved_negative.item(), 1)
        self.assertEqual(summary.n_pool_means_positive.item(), 1)
        self.assertEqual(summary.n_pool_means_negative.item(), 1)


if __name__ == "__main__":
    import unittest
    unittest.main()
