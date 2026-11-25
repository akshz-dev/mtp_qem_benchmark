"""Diagnostics quoted in the thesis that are derived from the stored datasets and predictions.

Usage (from the repository root): python code/thesis_diagnostics.py . [output.csv]
Writes results/thesis_diagnostics.csv with one row per quantity:

  zne_exp_fallback      rows where the exponential arm returned the linear ZNE value
                        (bench.zne_exp falls back when the fit fails, is non-finite or
                        extrapolates beyond 1.5 in magnitude); detected as stored
                        exponential prediction == stored linear prediction
  clip_sensitivity      Split A/B RMSE with predictions clipped to [-1, 1] instead of
                        [-1.5, 1.5], and the paired RBF-minus-fidelity difference
  gram_gaussian         similarity of the tuned fidelity Gram matrix (beta = 0.15,
                        Split A training rows) to a Gaussian kernel on the same
                        standardised features
  nonmonotone_error     share of rows whose shot-free |error| is not non-decreasing
                        across the three noise settings
  exact_input_zne       linear ZNE RMSE on shot-free inputs, all 4500 rows, Regime II
Needs only the stored files; nothing is re-simulated or refitted.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_models import qk_states, quantum_gram  # noqa: E402
from stats_util import paired_diff  # noqa: E402

FEATS = ["noisy_l1", "noisy_l2", "noisy_l3", "depth", "n_2q", "obs_weight"]
SEED = 7


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def zne_linear(values, grids):
    """Degree-one least-squares fit evaluated at zero, one grid per row."""
    return np.array([np.polynomial.polynomial.polyfit(g, v, 1)[0] for v, g in zip(values, grids)])


def main(root, out_path):
    rows = []

    def add(check, split, setting, value, n=None, note=""):
        rows.append({"check": check, "split": split, "setting": setting,
                     "value": value, "n": n, "note": note})

    data = pd.read_csv(os.path.join(root, "data", "benchmark_data.csv"))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
                  .split(data, groups=data["circuit_id"]))
    split_rows = {"A": te, "B": data.index[data.depth >= 5].to_numpy()}

    # Exponential-ZNE fallback and clipping sensitivity, from the stored predictions.
    for split, index in split_rows.items():
        stored = np.load(os.path.join(root, "results", f"predictions_split{split}.npz"))
        y, groups = stored["y"], data.loc[index, "circuit_id"].to_numpy()
        if not np.allclose(y, data.loc[index, "ideal"].to_numpy()):
            raise ValueError(f"Stored Split {split} rows are not in dataset order")
        exp, lin = stored["zne_exponential"], stored["zne_linear"]
        fallback = exp == lin
        add("zne_exp_fallback", split, "fallback_rows", int(fallback.sum()), len(y))
        add("zne_exp_fallback", split, "fallback_share", float(fallback.mean()), len(y))
        add("zne_exp_fallback", split, "exp_rmse_on_fitted_rows", rmse(y[~fallback], exp[~fallback]),
            int((~fallback).sum()))
        add("zne_exp_fallback", split, "linear_rmse_on_fitted_rows", rmse(y[~fallback], lin[~fallback]),
            int((~fallback).sum()))
        for model in sorted(m for m in stored.files if m != "y"):
            add("clip_sensitivity", split, f"{model}:clip_1.5", rmse(y, stored[model]), len(y))
            add("clip_sensitivity", split, f"{model}:clip_1.0", rmse(y, np.clip(stored[model], -1, 1)), len(y))
        for bound in (1.5, 1.0):
            pair = paired_diff(y, np.clip(stored["rbf_krr"], -bound, bound),
                               y, np.clip(stored["qk_ridge_quantum"], -bound, bound), groups, seed=SEED)
            for key in ("diff", "diff_lo", "diff_hi"):
                add("clip_sensitivity", split, f"rbf_minus_fidelity:clip_{bound}:{key}", pair[key], len(y))

    # Fidelity Gram matrix at the selected bandwidth against Gaussian kernels.
    choices = json.load(open(os.path.join(root, "results", "train_curves.json")))
    beta = choices["qk_selected_hyperparams"]["A_random_circuits"]["bandwidth"]
    rbf_gamma = choices["rbf_selected_hyperparams"]["A_random_circuits"]["gamma"]
    features = StandardScaler().fit_transform(data.loc[tr, FEATS].to_numpy())
    fidelity = quantum_gram(qk_states(features, beta))
    distances = euclidean_distances(features, squared=True)
    off = ~np.eye(len(features), dtype=bool)
    q = fidelity[off]

    def compare(gamma):
        g = np.exp(-gamma * distances[off])
        return float(np.sqrt(np.mean((q - g) ** 2))), float(np.corrcoef(q, g)[0, 1])

    scan = [(gamma, *compare(gamma)) for gamma in np.geomspace(1e-3, 1.0, 200)]
    best_gamma, best_rms, best_corr = min(scan, key=lambda t: t[1])
    add("gram_gaussian", "A", "fidelity_bandwidth", beta, len(features))
    add("gram_gaussian", "A", "fidelity_offdiag_mean", float(q.mean()), int(q.size))
    add("gram_gaussian", "A", "best_gaussian_gamma", float(best_gamma), int(q.size),
        "gamma minimising the RMS entry difference over 200 log-spaced values in [1e-3, 1]")
    add("gram_gaussian", "A", "best_gaussian_rms_entry_diff", best_rms, int(q.size))
    add("gram_gaussian", "A", "best_gaussian_correlation", best_corr, int(q.size))
    for gamma in (0.005, rbf_gamma):
        rms, corr = compare(gamma)
        add("gram_gaussian", "A", f"gaussian_gamma_{gamma:g}_rms_entry_diff", rms, int(q.size))
        add("gram_gaussian", "A", f"gaussian_gamma_{gamma:g}_correlation", corr, int(q.size))

    # Shot-free error against noise setting, and linear ZNE on shot-free inputs.
    def nonmonotone(frame, columns):
        err = np.abs(frame[columns].to_numpy() - frame.ideal.to_numpy()[:, None])
        return float(np.mean(~((err[:, 0] <= err[:, 1]) & (err[:, 1] <= err[:, 2]))))

    param = ["exact_noisy_l1", "exact_noisy_l2", "exact_noisy_l3"]
    fold = ["exact_fold_l1", "exact_fold_l2", "exact_fold_l3"]
    ratios = ["fold_ratio_l1", "fold_ratio_l2", "fold_ratio_l3"]
    add("nonmonotone_error", "all", "regime_I:parameter", nonmonotone(data, param), len(data))
    labels = {"A": "production", "M": "defect_removed", "R": "defect_reset"}
    for key, label in labels.items():
        frame = pd.read_csv(os.path.join(root, "data", f"ds_real_{key}.csv"))
        add("nonmonotone_error", "all", f"regime_II_{label}:parameter", nonmonotone(frame, param), len(frame))
        add("nonmonotone_error", "all", f"regime_II_{label}:folding", nonmonotone(frame, fold), len(frame))
        y = frame.ideal.to_numpy()
        par_grid = np.broadcast_to([1.0, 2.0, 3.0], (len(frame), 3))
        add("exact_input_zne", "all", f"regime_II_{label}:unmitigated", rmse(y, frame.exact_noisy_l1), len(frame))
        add("exact_input_zne", "all", f"regime_II_{label}:linear_parameter",
            rmse(y, zne_linear(frame[param].to_numpy(), par_grid)), len(frame))
        add("exact_input_zne", "all", f"regime_II_{label}:linear_folding",
            rmse(y, zne_linear(frame[fold].to_numpy(), frame[ratios].to_numpy())), len(frame))

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "."
    main(project, sys.argv[2] if len(sys.argv) > 2 else os.path.join(project, "results", "thesis_diagnostics.csv"))
