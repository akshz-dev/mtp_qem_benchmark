"""Regression checks for ZNE with circuit-specific executed fold scales."""

from unittest import TestCase, mock

import numpy as np
import pandas as pd

import bench
import experiments
from experiments_realistic import FOLD_FEATS, PARAM_FEATS, as_param
from simulate_realistic import FOLD_RATIO_COLUMNS, LAMBDAS, fold_pattern, folded_layer_ratio


class FoldedZNETests(TestCase):
    def setUp(self):
        self.scales = np.array([[1, 7 / 3, 3], [1, 9 / 5, 3], [1, 2, 3], [1, 7 / 3, 3]])
        self.intercept = np.array([0.2, -0.3, 0.4, 0.1])
        self.slope = np.array([0.05, 0.1, -0.04, -0.06])
        self.linear = self.intercept[:, None] + self.slope[:, None] * self.scales

    def test_nominal_grid_retains_existing_polynomial_path(self):
        for degree in (1, 2):
            with self.subTest(degree=degree):
                expected = np.polynomial.polynomial.polyfit(bench.LAMS, self.linear.T, degree)[0]
                np.testing.assert_array_equal(bench.zne_poly(self.linear, degree), expected)
                np.testing.assert_array_equal(
                    bench.zne_poly(self.linear, degree, np.broadcast_to(bench.LAMS, self.linear.shape)),
                    expected,
                )

    def test_polynomial_uses_executed_grid_for_each_observation(self):
        np.testing.assert_allclose(bench.zne_poly(self.linear, 1, self.scales), self.intercept, atol=1e-12)
        quadratic = self.linear + 0.02 * self.scales ** 2
        np.testing.assert_allclose(bench.zne_poly(quadratic, 2, self.scales), self.intercept, atol=1e-12)
        self.assertFalse(np.allclose(bench.zne_poly(quadratic, 2), self.intercept))

    def test_exponential_fit_and_fallback_use_the_same_actual_grid(self):
        values = 0.6 * np.exp(-0.5 * self.scales) - 0.1
        np.testing.assert_allclose(bench.zne_exp(values, self.scales), 0.5, atol=1e-8)

        for failure in ({"side_effect": RuntimeError("failed fit")},
                        {"return_value": (np.array([2.0, 0.5, 0.0]), None)}):
            with self.subTest(failure=failure):
                with mock.patch.object(bench, "curve_fit", **failure) as curve_fit:
                    np.testing.assert_allclose(bench.zne_exp(self.linear, self.scales), self.intercept, atol=1e-12)
                actual_grids = [call.args[1] for call in curve_fit.call_args_list]
                np.testing.assert_array_equal(actual_grids, self.scales)

    def test_run_arms_routes_grid_to_zne_without_changing_learned_inputs(self):
        frame = pd.DataFrame(self.linear, columns=experiments.LAM_FEATS)
        frame["depth"] = [3, 5, 4, 3]
        frame["n_2q"] = frame.depth * 5
        frame["obs_weight"] = 1
        frame["circuit_id"] = np.arange(len(frame))
        frame["ideal"] = self.intercept
        frame[experiments.ZNE_SCALE_COLUMNS] = self.scales
        train, test = [1, 2], [0, 3]
        learned_model = mock.Mock(scale=0.15, alpha=0.1, cv_r2=0.5, n_candidates=24)
        learned_model.predict.return_value = np.zeros(len(test))

        with mock.patch.object(experiments, "fit_kernel_ridge", return_value=learned_model) as fit:
            preds, _, _, _ = experiments.run_arms(
                frame, train, test, arms=("zne_linear", "zne_richardson2", "rbf_krr"), fast=True,
            )

        np.testing.assert_allclose(preds["zne_linear"], self.intercept[test], atol=1e-12)
        np.testing.assert_allclose(preds["zne_richardson2"], self.intercept[test], atol=1e-12)
        np.testing.assert_array_equal(fit.call_args.args[1], frame.loc[train, bench.FEATS].to_numpy())
        np.testing.assert_array_equal(
            learned_model.predict.call_args.args[0], frame.loc[test, bench.FEATS].to_numpy(),
        )

    def test_realistic_folding_uses_recorded_ratios_and_shared_native_measurement(self):
        frame = pd.DataFrame(self.linear, columns=experiments.LAM_FEATS)
        frame["depth"] = [3, 5, 4, 3]
        frame["n_2q"] = frame.depth * 5
        frame["obs_weight"] = 1
        frame[FOLD_FEATS[:3]] = self.linear + [0, 0.1, 0.2]
        frame[list(FOLD_RATIO_COLUMNS)] = [
            [folded_layer_ratio(fold_pattern(depth, scale)) for scale in LAMBDAS]
            for depth in frame.depth
        ]

        folded = as_param(frame, FOLD_FEATS)
        np.testing.assert_allclose(folded[experiments.ZNE_SCALE_COLUMNS], self.scales, atol=1e-14)
        np.testing.assert_array_equal(folded[experiments.LAM_FEATS], frame[FOLD_FEATS[:3]])
        self.assertTrue(folded["noisy_l1"].equals(frame["noisy_l1"]))
        self.assertTrue(as_param(frame, PARAM_FEATS).equals(frame))

        frame.loc[0, "fold_l1"] += 0.1
        with self.assertRaisesRegex(ValueError, "share the native shot estimate"):
            as_param(frame, FOLD_FEATS)
