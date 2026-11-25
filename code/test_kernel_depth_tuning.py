"""Checks for forward-depth grouped tuning of both kernel regressors."""
from unittest import TestCase

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from kernel_models import KERNEL_MODELS, fit_kernel_ridge, forward_depth_splits, qk_states, quantum_gram


class KernelDepthTuningTests(TestCase):
    def setUp(self):
        self.depths = np.repeat((2, 3, 4), 8)
        self.groups = np.repeat(np.arange(12), 2)
        rng = np.random.default_rng(7)
        self.X = np.column_stack((rng.uniform(-1, 1, size=(24, 3)), self.depths))
        self.y = 0.6 * self.X[:, 0] + 0.2 * self.X[:, 1] ** 2

    def test_all_validation_rows_are_deeper_and_circuit_groups_stay_whole(self):
        folds = forward_depth_splits(self.depths, self.groups)
        self.assertEqual(len(folds), 4)
        for (tr, va), depth in zip(folds, (3, 3, 4, 4), strict=True):
            np.testing.assert_array_equal(tr, np.flatnonzero(self.depths < depth))
            np.testing.assert_array_equal(self.depths[va], depth)
            groups = np.unique(self.groups[va])
            np.testing.assert_array_equal(va, np.flatnonzero(np.isin(self.groups, groups)))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([va for _, va in folds])),
            np.flatnonzero(self.depths >= 3),
        )

    def test_held_out_depth_and_cross_depth_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outer training depths"):
            forward_depth_splits(np.r_[self.depths, 5], np.r_[self.groups, 12])
        groups = self.groups.copy()
        groups[8] = groups[0]
        with self.assertRaisesRegex(ValueError, "single depth"):
            forward_depth_splits(self.depths, groups)

    def test_matched_kernel_fold_scores_and_refit(self):
        folds = forward_depth_splits(self.depths, self.groups)
        for model in KERNEL_MODELS:
            with self.subTest(model=model):
                fitted = fit_kernel_ridge(
                    model, self.X, self.y, self.groups, fast=True,
                    cv_splits=folds, return_diagnostics=True,
                )
                diag = fitted.diagnostics
                self.assertEqual(diag.fold_scores.shape, (4, 3, 8))
                self.assertEqual(fitted.n_candidates, 24)
                self.assertEqual(diag.scales[diag.selected_index[0]], fitted.scale)
                self.assertEqual(diag.alphas[diag.selected_index[1]], fitted.alpha)
                np.testing.assert_allclose(diag.mean_scores, diag.fold_scores.mean(axis=0))
                self.assertEqual(fitted.cv_r2, diag.mean_scores.max())

                scale, alpha = fitted.scale, fitted.alpha
                for fold, (tr, va) in enumerate(folds):
                    scaler = StandardScaler().fit(self.X[tr])
                    train, valid = scaler.transform(self.X[tr]), scaler.transform(self.X[va])
                    if model == "rbf_krr":
                        gram = rbf_kernel(train, gamma=scale)
                        cross = rbf_kernel(valid, train, gamma=scale)
                    else:
                        states = qk_states(train, scale)
                        gram = quantum_gram(states)
                        cross = quantum_gram(qk_states(valid, scale), states)
                    estimates = cross @ np.linalg.solve(gram + alpha * np.eye(len(tr)), self.y[tr])
                    actual = self.y[va]
                    expected_r2 = 1 - np.sum((actual - estimates) ** 2) / np.sum(
                        (actual - actual.mean()) ** 2
                    )
                    self.assertAlmostEqual(diag.fold_scores[(fold, *diag.selected_index)], expected_r2)

                self.assertEqual(len(fitted.coefficients), len(self.X))
                np.testing.assert_allclose(fitted.scaler.mean_, self.X.mean(axis=0))
