"""Paired, circuit-grouped bootstrap for differences between two error estimates.

Two methods scored on the SAME test rows are correlated, so whether their
marginal bootstrap intervals overlap is not a test of whether they differ.
paired_diff resamples test circuits (all rows of a circuit together) and
recomputes RMSE(a) - RMSE(b) on each resample. The difference is resolved at
the 95% level when its interval excludes zero.
"""
import numpy as np


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def paired_diff(y_a, p_a, y_b, p_b, groups, n_boot=1000, seed=0):
    """RMSE(p_a vs y_a) - RMSE(p_b vs y_b) with a grouped paired bootstrap CI.

    y_a and y_b may differ (e.g. the same circuits under two noise configurations
    share ideal values, so they are equal; a drift comparison has equal targets
    too). Rows must be aligned: row i of a and row i of b belong to the same
    circuit and observable.
    """
    y_a, p_a, y_b, p_b, groups = map(np.asarray, (y_a, p_a, y_b, p_b, groups))
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uq}
    d = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(uq, size=len(uq), replace=True)])
        d[k] = _rmse(y_a[idx], p_a[idx]) - _rmse(y_b[idx], p_b[idx])
    point = _rmse(y_a, p_a) - _rmse(y_b, p_b)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": point, "diff_lo": float(lo), "diff_hi": float(hi),
            "resolved": bool(lo > 0 or hi < 0)}
