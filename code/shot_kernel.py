"""Finite-shot fidelity kernels with train-only spectral repair."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import eigh
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from kernel_models import qk_states, quantum_gram


BANDWIDTH = 0.15
N_FOLDS = 4
SHOT_BUDGETS = (100, 1000, 10000, 100000, None)
FINE_ALPHAS = np.logspace(-6, 1, 23)
MAIN_GRID = slice(None, None, 2)  # The original 12 points in the nested grid.
SHOT_PROTOCOL = "fold_local_scaling_psd_projection"


@dataclass
class PositiveSpectrum:
    """Retained eigenspace of one measured training Gram matrix."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    negative_eigenvalues: int
    min_eigenvalue: float

    @classmethod
    def fit(cls, gram):
        values, vectors = eigh(gram, driver="evd", check_finite=False)
        # Treat numerical nulls as zero, including small positive roundoff.
        tolerance = len(values) * np.finfo(values.dtype).eps * np.max(np.abs(values))
        keep = values > tolerance
        return cls(values[keep], vectors[:, keep], int(np.count_nonzero(values < -tolerance)),
                   float(values[0]))

    def predict(self, cross, targets, alphas):
        """Predict using K_cross V+ and the same retained ridge eigenspace."""
        coefficients = (self.vectors.T @ targets)[:, None] / (self.eigenvalues[:, None] + alphas)
        return (cross @ self.vectors) @ coefficients


@dataclass
class FidelityProbabilities:
    """Fidelity probabilities after fitting the scaler on the block's training rows."""

    train: np.ndarray
    cross: np.ndarray
    train_indices: np.ndarray
    eval_indices: np.ndarray

    @property
    def n_entries(self):
        n_train = len(self.train)
        return n_train * (n_train + 1) // 2 + self.cross.size

    @cached_property
    def triangle(self):
        return np.triu_indices(len(self.train))

    @cached_property
    def train_probabilities(self):
        return np.clip(self.train[self.triangle], 0, 1)

    @cached_property
    def cross_probabilities(self):
        return np.clip(self.cross, 0, 1)

    def sample(self, shots, rng):
        if shots is None:
            return self.train, self.cross
        values = rng.binomial(shots, self.train_probabilities) / shots
        gram = np.empty_like(self.train)
        row, col = self.triangle
        gram[row, col] = values
        gram[col, row] = values
        return gram, rng.binomial(shots, self.cross_probabilities) / shots


def fit_grouped_block(train_features, eval_features, bandwidth=BANDWIDTH, *,
                      train_indices=None, eval_indices=None):
    scaler = StandardScaler()
    train = qk_states(scaler.fit_transform(train_features), bandwidth)
    evaluate = qk_states(scaler.transform(eval_features), bandwidth)
    return FidelityProbabilities(
        quantum_gram(train), quantum_gram(evaluate, train),
        np.arange(len(train)) if train_indices is None else train_indices,
        np.arange(len(evaluate)) if eval_indices is None else eval_indices,
    )


def fit_probability_blocks(X_train, X_test, groups, bandwidth=BANDWIDTH):
    """Build only train-to-train and evaluation-to-train blocks."""
    folds = [fit_grouped_block(X_train[tr], X_train[va], bandwidth, train_indices=tr, eval_indices=va)
             for tr, va in GroupKFold(n_splits=N_FOLDS).split(X_train, groups=groups)]
    return folds, fit_grouped_block(X_train, X_test, bandwidth)


def _draw_rng(seed, shots, draw, block):
    return np.random.default_rng(np.random.SeedSequence([seed, shots or 0, draw, block]))


def _validation_scores(job):
    fold, block, targets, draws, seed = job
    train_targets = targets[block.train_indices]
    eval_targets = targets[block.eval_indices]
    variance = np.sum((eval_targets - eval_targets.mean()) ** 2)
    scores = []
    with threadpool_limits(limits=1):
        for shots, draw in draws:
            gram, cross = block.sample(shots, _draw_rng(seed, shots, draw, fold))
            predicted = PositiveSpectrum.fit(gram).predict(cross, train_targets, FINE_ALPHAS)
            error = np.sum((eval_targets[:, None] - predicted) ** 2, axis=0)
            scores.append(1 - error / variance if variance else (error == 0).astype(float))
    print(f"    grouped fold {fold + 1}: {len(draws)} draws scored", flush=True)
    return np.array(scores)


def _final_predictions(job):
    block, train_targets, selections, seed = job
    fitted = []
    with threadpool_limits(limits=1):
        for shots, draw, alpha, fine_alpha in selections:
            gram, cross = block.sample(shots, _draw_rng(seed, shots, draw, N_FOLDS))
            spectrum = PositiveSpectrum.fit(gram)
            predictions = spectrum.predict(cross, train_targets, np.array([alpha, fine_alpha]))
            fitted.append((predictions, spectrum.eigenvalues.size, spectrum.negative_eigenvalues,
                           spectrum.min_eigenvalue))
            print(f"    full-training fit: shots={shots or 'exact'} draw={draw}", flush=True)
    return fitted


def run_shot_study(X_train, y_train, groups, X_test, shots_list=SHOT_BUDGETS, *,
                   seed=7, n_draws=5, workers=1):
    """Select penalties from grouped CV, then refit and predict without test labels."""
    if workers < 1 or n_draws < 1:
        raise ValueError("workers and n_draws must be positive")
    draws = [(shots, draw) for shots in shots_list for draw in range(1 if shots is None else n_draws)]
    with threadpool_limits(limits=1):
        folds, final = fit_probability_blocks(X_train, X_test, groups)
    cv_entries = sum(block.n_entries for block in folds)
    jobs = [(i, block, y_train, draws, seed) for i, block in enumerate(folds)]
    if workers == 1:
        scores = np.mean(list(map(_validation_scores, jobs)), axis=0)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, N_FOLDS)) as pool:
            scores = np.mean(list(pool.map(_validation_scores, jobs)), axis=0)
    main_indices = 2 * scores[:, MAIN_GRID].argmax(axis=1)
    fine_indices = scores.argmax(axis=1)
    selections = [(shots, draw, FINE_ALPHAS[main], FINE_ALPHAS[fine])
                  for (shots, draw), main, fine in zip(draws, main_indices, fine_indices)]

    # On the outer holdout, use a new spectrum fitted on the full training
    # Gram. Adjacent batches share its sampling probabilities, not its noise.
    if workers == 1:
        fitted = _final_predictions((final, y_train, selections, seed))
    else:
        n_batches = min(workers, len(draws))
        batches = [(final, y_train, selections[i::n_batches], seed) for i in range(n_batches)]
        with ProcessPoolExecutor(max_workers=n_batches) as pool:
            fitted = [None] * len(draws)
            for i, batch in enumerate(pool.map(_final_predictions, batches)):
                fitted[i::n_batches] = batch

    return [
        {
            "shots_per_entry": shots or 0,
            "draw": draw,
            "kernel_alpha": alpha,
            "kernel_cv_r2": scores[i, main_indices[i]],
            "kernel_candidates": len(FINE_ALPHAS[MAIN_GRID]),
            "fine_alpha": fine_alpha,
            "fine_cv_r2": scores[i, fine_indices[i]],
            "fine_candidates": len(FINE_ALPHAS),
            "psd_rank": rank,
            "negative_eigenvalues": negative,
            "min_eigenvalue": min_eigenvalue,
            "cv_circuit_executions": int(shots or 0) * cv_entries,
            "final_circuit_executions": int(shots or 0) * final.n_entries,
            "shot_protocol": SHOT_PROTOCOL,
            "predictions": predictions,
        }
        for i, ((shots, draw, alpha, fine_alpha), (predictions, rank, negative, min_eigenvalue))
        in enumerate(zip(selections, fitted))
    ]
