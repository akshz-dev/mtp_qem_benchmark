"""Regression checks for grouped finite-shot kernel repair and prediction."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from kernel_models import qk_states, quantum_gram
from shot_kernel import PositiveSpectrum, fit_probability_blocks, run_shot_study


class ShotKernelTests(unittest.TestCase):
    def test_predict_matches_explicit_psd_kernel_and_cross_projection(self):
        gram = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        cross = np.array([[0.65, 0.40, 0.55], [0.15, 0.65, 0.90]])
        y = np.array([-0.90, 0.50, 0.70])
        alphas = np.array([1e-6, 0.04, 2.0])

        # This Gram has one negative eigenvalue, 1 - sqrt(2). Remove it
        # analytically rather than reusing the production eigendecomposition.
        discarded = np.array([0.5, -np.sqrt(0.5), 0.5])
        discarded_projection = np.outer(discarded, discarded)
        retained_projection = np.eye(len(y)) - discarded_projection
        repaired_train = gram - (1.0 - np.sqrt(2.0)) * discarded_projection
        projected_cross = cross @ retained_projection
        expected = np.column_stack([
            projected_cross @ np.linalg.solve(repaired_train + alpha * np.eye(len(y)), y)
            for alpha in alphas
        ])

        repair = PositiveSpectrum.fit(gram)
        prediction = repair.predict(cross, y, alphas)
        assert_allclose(prediction, expected, atol=2e-9, rtol=2e-9)
        # A discarded component must stay discarded for validation and test.
        perturbed_cross = cross + np.array([[0.10], [-0.05]]) * discarded
        assert_allclose(repair.predict(perturbed_cross, y, alphas), prediction, atol=1e-12)

    def test_kernel_blocks_use_only_fold_training_circuits_to_fit_scaling(self):
        groups = np.repeat(np.arange(6), 2)
        x = np.column_stack([np.arange(12), np.cos(np.arange(12))])
        heldout = np.array([[-15.0, 2.0], [18.0, 3.0], [22.0, -2.0]])
        bandwidth = 0.15
        folds, final = fit_probability_blocks(x, heldout, groups, bandwidth=bandwidth)
        splits = list(GroupKFold(n_splits=4).split(x, groups=groups))

        def assert_fidelity_blocks(block, train, validation):
            scaler = StandardScaler().fit(train)
            train_states = qk_states(scaler.transform(train), bandwidth)
            eval_states = qk_states(scaler.transform(validation), bandwidth)
            assert_allclose(block.train, quantum_gram(train_states), atol=1e-12)
            assert_allclose(block.cross, quantum_gram(eval_states, train_states), atol=1e-12)
            self.assertEqual(block.cross.shape, (len(validation), len(train)))

        for fold, (train, validation) in zip(folds, splits, strict=True):
            assert_array_equal(fold.train_indices, train)
            assert_array_equal(fold.eval_indices, validation)
            self.assertEqual(len(np.intersect1d(groups[train], groups[validation])), 0)
            assert_fidelity_blocks(fold, x[train], x[validation])

        assert_fidelity_blocks(final, x, heldout)
        assert_array_equal(final.train_indices, np.arange(len(x)))
        assert_array_equal(final.eval_indices, np.arange(len(heldout)))

        # Perturb a fold's validation features without changing its training
        # features. Both the training scaler and training probabilities are fixed.
        changed = x.copy()
        changed[folds[0].eval_indices] += [90.0, -25.0]
        new_folds, _ = fit_probability_blocks(changed, heldout, groups, bandwidth=bandwidth)
        assert_allclose(new_folds[0].train, folds[0].train, atol=1e-12)
        self.assertFalse(np.allclose(new_folds[0].cross, folds[0].cross))

    def test_shot_study_selects_and_predicts_on_projected_blocks(self):
        groups = np.repeat(np.arange(6), 2)
        x = np.column_stack([np.arange(12), np.cos(np.arange(12))])
        y = np.sin(x[:, 0] / 3) + 0.2 * x[:, 1]
        heldout = np.array([[-1.0, 0.5], [11.0, -0.5]])
        seed, shots = 17, 25
        alphas = np.logspace(-6, 1, 23)
        folds, final = fit_probability_blocks(x, heldout, groups)

        def explicit_predictions(block, block_index, penalties):
            rng = np.random.default_rng(np.random.SeedSequence([seed, shots, 0, block_index]))
            gram, cross = block.sample(shots, rng)
            values, basis = np.linalg.eigh(gram)
            keep = values > len(values) * np.finfo(values.dtype).eps * max(abs(values))
            positive = basis[:, keep]
            repaired = (positive * values[keep]) @ positive.T
            projected = (cross @ positive) @ positive.T
            return np.column_stack([
                projected @ np.linalg.solve(repaired + alpha * np.eye(len(gram)), y[block.train_indices])
                for alpha in penalties
            ])

        scores = []
        for i, fold in enumerate(folds):
            targets = y[fold.eval_indices]
            variance = np.sum((targets - targets.mean()) ** 2)
            errors = explicit_predictions(fold, i, alphas) - targets[:, None]
            scores.append(1 - np.sum(errors ** 2, axis=0) / variance)
        scores = np.mean(scores, axis=0)
        selected = [2 * scores[::2].argmax(), scores.argmax()]

        [result] = run_shot_study(x, y, groups, heldout, (shots,), seed=seed, n_draws=1)
        assert_allclose([result["kernel_alpha"], result["fine_alpha"]], alphas[selected])
        assert_allclose([result["kernel_cv_r2"], result["fine_cv_r2"]], scores[selected], atol=1e-8)
        expected = explicit_predictions(final, len(folds), alphas[selected])
        assert_allclose(result["predictions"], expected, atol=1e-8)
        cv_entries = sum(n * (n + 1) // 2 + n * (len(x) - n) for n in (8, 8, 10, 10))
        self.assertEqual(result["cv_circuit_executions"], shots * cv_entries)
        self.assertEqual(result["final_circuit_executions"], shots * (12 * 13 // 2 + 12 * 2))

        changed_holdout = heldout + [30.0, -5.0]
        [changed] = run_shot_study(x, y, groups, changed_holdout, (shots,), seed=seed, n_draws=1)
        for field in ("kernel_alpha", "fine_alpha", "kernel_cv_r2", "fine_cv_r2"):
            self.assertEqual(changed[field], result[field])
        self.assertFalse(np.allclose(changed["predictions"], result["predictions"]))


if __name__ == "__main__":
    unittest.main()
