"""Matched classical RBF and exact IQP fidelity kernel ridge regression."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.linalg import eigh, solve
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


QK_BANDWIDTHS = (0.05, 0.15, 0.4, 1.0)
RBF_GAMMAS = (0.005, 0.02, 0.08, 0.32)
KERNEL_MODELS = ("qk_ridge_quantum", "rbf_krr")


@lru_cache(maxsize=None)
def _iqp_signs(n_features):
    """Z and pairwise ZZ eigenvalues in PennyLane's wire order."""
    states = np.arange(1 << n_features)[:, None]
    z = 1.0 - 2 * ((states >> np.arange(n_features - 1, -1, -1)) & 1)
    left, right = np.asarray(list(combinations(range(n_features), 2)), dtype=int).reshape(-1, 2).T
    return z.T, z[:, left].T * z[:, right].T, left, right


def qk_states(X, bw):
    """Batched exact states of two all-to-all IQPEmbedding repetitions."""
    angles = bw * np.asarray(X)
    z, zz, left, right = _iqp_signs(angles.shape[1])
    phase = np.exp(-0.5j * (angles @ z + (angles[:, left] * angles[:, right]) @ zz))
    # The first H layer sends |0...0> to the uniform state. The second
    # H layer is a Walsh-Hadamard transform of the first diagonal phase.
    n_samples, n_states = phase.shape
    state = phase.copy()
    for power in range(angles.shape[1]):
        block = state.reshape(n_samples, -1, 2, 1 << power)
        low = block[:, :, 0].copy()
        high = block[:, :, 1]
        block[:, :, 0] = low + high
        block[:, :, 1] = low - high
    return state * phase / n_states


def quantum_gram(Psi_a, Psi_b=None):
    Psi_b = Psi_a if Psi_b is None else Psi_b
    return np.abs(Psi_a @ Psi_b.conj().T) ** 2


def forward_depth_splits(depths, groups):
    """Validate at each of depths 3 and 4 using two circuit-grouped subsets."""
    depths, groups = np.asarray(depths), np.asarray(groups)
    if depths.ndim != 1 or groups.shape != depths.shape:
        raise ValueError("Expected one depth and circuit group per training row")
    if not np.array_equal(np.unique(depths), [2, 3, 4]):
        raise ValueError("Forward-depth tuning requires outer training depths 2, 3 and 4")
    group_counts = sum(len(np.unique(groups[depths == depth])) for depth in (2, 3, 4))
    if group_counts != len(np.unique(groups)):
        raise ValueError("Each circuit group must belong to a single depth")

    folds = []
    for depth in (3, 4):
        train = np.flatnonzero(depths < depth)
        validation = np.flatnonzero(depths == depth)
        splitter = GroupKFold(n_splits=2)
        for _, subset in splitter.split(validation, groups=groups[validation]):
            folds.append((train, validation[subset]))
    return tuple(folds)


@dataclass
class KernelCVDiagnostics:
    scales: tuple[float, ...]
    alphas: np.ndarray
    fold_scores: np.ndarray
    mean_scores: np.ndarray
    selected_index: tuple[int, int]


@dataclass
class FittedKernelRidge:
    model: str
    scale: float
    alpha: float
    cv_r2: float
    n_candidates: int
    scaler: StandardScaler
    training_representation: np.ndarray
    coefficients: np.ndarray
    diagnostics: KernelCVDiagnostics | None = None

    def predict(self, X):
        X = self.scaler.transform(X)
        if self.model == "rbf_krr":
            gram = rbf_kernel(X, self.training_representation, gamma=self.scale)
        else:
            gram = quantum_gram(qk_states(X, self.scale), self.training_representation)
        return gram @ self.coefficients


def _ridge_path(gram, target, alphas):
    """Compute ridge coefficients for all alphas with one eigendecomposition."""
    eigenvalues, vectors = eigh(gram, driver="evd", check_finite=False, overwrite_a=True)
    # Both exact kernels are PSD. Small negative eigenvalues are roundoff.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    spectral_coeffs = (vectors.T @ target)[:, None] / (eigenvalues[:, None] + alphas)
    return vectors @ spectral_coeffs


def _score_alphas(cross_gram, coefficients, validation):
    predicted = cross_gram @ coefficients
    error = np.sum((validation[:, None] - predicted) ** 2, axis=0)
    variance = np.sum((validation - validation.mean()) ** 2)
    return 1 - error / variance if variance else (error == 0).astype(float)


def _checked_splits(cv_splits, groups):
    folds = []
    for tr, va in cv_splits:
        tr, va = np.asarray(tr), np.asarray(va)
        for indices in (tr, va):
            if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
                    or indices.min() < 0 or indices.max() >= len(groups)
                    or np.unique(indices).size != indices.size):
                raise ValueError("Each tuning fold needs unique, valid row indices")
        if np.intersect1d(groups[tr], groups[va]).size:
            raise ValueError("A tuning fold splits a circuit group")
        folds.append((tr, va))
    if not folds:
        raise ValueError("At least one tuning fold is required")
    return folds


def fit_kernel_ridge(model, X_train, y_train, groups, *, fast=False, fixed_scale=None,
                     cv_splits=None, return_diagnostics=False):
    """Tune on group folds, then refit the selected candidate on all training rows."""
    if model not in KERNEL_MODELS:
        raise ValueError(f"Unknown kernel ridge model: {model}")
    is_rbf = model == "rbf_krr"
    scales = RBF_GAMMAS if is_rbf else QK_BANDWIDTHS
    if fast:
        scales = scales[:3]
    if fixed_scale is not None:
        scales = (fixed_scale,)
    alphas = np.logspace(-6, 1, 8 if fast else 15)
    X_train, y_train, groups = map(np.asarray, (X_train, y_train, groups))
    if groups.shape != (len(X_train),):
        raise ValueError("Expected one circuit group per training row")
    if cv_splits is None:
        cv_splits = GroupKFold(n_splits=4).split(X_train, y_train, groups)
    folds = _checked_splits(cv_splits, groups)
    fold_scores = np.empty((len(folds), len(scales), len(alphas)))
    # Both validation subsets at a depth use the same strictly shallower training rows.
    by_training_rows = {}
    for fold, (tr, va) in enumerate(folds):
        by_training_rows.setdefault(tr.astype(np.intp, copy=False).tobytes(), []).append((fold, tr, va))
    for validation_folds in by_training_rows.values():
        tr = validation_folds[0][1]
        scaler = StandardScaler()
        train = scaler.fit_transform(X_train[tr])
        validation = [(fold, va, scaler.transform(X_train[va])) for fold, _, va in validation_folds]
        if is_rbf:
            train_distances = euclidean_distances(train, squared=True)
            validation = [(fold, va, euclidean_distances(valid, train, squared=True))
                          for fold, va, valid in validation]
        for i, scale in enumerate(scales):
            if is_rbf:
                gram = np.exp(-scale * train_distances)
            else:
                states = qk_states(train, scale)
                gram = quantum_gram(states)
            coefficients = _ridge_path(gram, y_train[tr], alphas)
            for fold, va, valid in validation:
                if is_rbf:
                    cross_gram = np.exp(-scale * valid)
                else:
                    cross_gram = quantum_gram(qk_states(valid, scale), states)
                fold_scores[fold, i] = _score_alphas(cross_gram, coefficients, y_train[va])

    scores = np.zeros(fold_scores.shape[1:])
    for scores_by_fold in fold_scores:
        scores += scores_by_fold
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    mean_scores = scores / len(folds)
    diagnostics = None
    if return_diagnostics:
        diagnostics = KernelCVDiagnostics(scales, alphas, fold_scores, mean_scores, (int(i), int(j)))
    scale, alpha = scales[i], float(alphas[j])
    scaler = StandardScaler()
    train = scaler.fit_transform(X_train)
    if is_rbf:
        representation = train
        gram = rbf_kernel(train, gamma=scale)
    else:
        representation = qk_states(train, scale)
        gram = quantum_gram(representation)
    gram.flat[::len(train) + 1] += alpha
    coefficients = solve(gram, y_train, assume_a="pos", check_finite=False, overwrite_a=True)
    return FittedKernelRidge(model, scale, alpha, float(mean_scores[i, j]), scores.size,
                            scaler, representation, coefficients, diagnostics)
