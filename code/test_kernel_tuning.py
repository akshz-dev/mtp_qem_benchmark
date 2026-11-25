"""Checks for the circuit-grouped tuning of both kernel regressors."""
from unittest import TestCase

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from kernel_models import KERNEL_MODELS, fit_kernel_ridge, qk_states, quantum_gram


class KernelTuningTests(TestCase):
    def setUp(self):
        self.depths = np.repeat((2, 3, 4), 8)
        self.groups = np.repeat(np.arange(12), 2)
        rng = np.random.default_rng(7)
        self.X = np.column_stack((rng.uniform(-1, 1, size=(24, 3)), self.depths))
        self.y = 0.6 * self.X[:, 0] + 0.2 * self.X[:, 1] ** 2

    def test_split_circuit_groups_are_rejected(self):
        folds = list(GroupKFold(n_splits=4).split(self.X, groups=self.groups))
        tr, va = folds[0]
        leaky = [(np.r_[tr, va[:1]], va)]
        with self.assertRaisesRegex(ValueError, "splits a circuit group"):
            fit_kernel_ridge("rbf_krr", self.X, self.y, self.groups, fast=True, cv_splits=leaky)

    def test_matched_kernel_fold_scores_and_refit(self):
        folds = list(GroupKFold(n_splits=4).split(self.X, groups=self.groups))
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
                    self.assertFalse(set(self.groups[tr]) & set(self.groups[va]))
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


if __name__ == "__main__":
    import unittest
    unittest.main()
