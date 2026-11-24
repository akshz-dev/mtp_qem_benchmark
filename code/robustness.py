"""Robustness of the headline comparisons to the train/test split.

Every result in the main benchmark uses one circuit-grouped 70/30 split (seed 7).
This script repeats the comparison over N_SPLITS independent circuit-grouped
splits and replaces "do the two bootstrap intervals overlap?" with a paired,
circuit-grouped bootstrap on the RMSE DIFFERENCE between two methods evaluated
on the same test circuits (stats_util.paired_diff).

Usage: python robustness.py --config ideal|real_param|real_fold|real_reset_param|real_reset_fold

What varies per repetition: the train/test split, the cross-validation folds
that follow from it, and the scikit-learn model seeds (MLP initialisation,
tree subsampling). What does not vary: the dataset, and the variational-circuit
initialisation seed (bound at definition time in bench.py / experiments.py).
Outputs: robust_splits_<config>.csv (one row per split x arm) and
robust_pairs_<config>.csv (paired differences against three reference arms).
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import experiments as ex
from bench import FEATS, rmse
from stats_util import paired_diff

N_SPLITS = 10
N_BOOT = 1000
ARMS = ["no_mitigation", "zne_linear", "zne_richardson2", "zne_exponential",
        "ridge", "hist_gbm", "mlp", "qk_ridge_quantum", "dr_vqc_quantum",
        "bounded_vqc_quantum"]
REFS = ["no_mitigation", "zne_linear", "mlp"]
FOLD = ["fold_l1", "fold_l2", "fold_l3"]
CONFIGS = {
    "ideal": ("benchmark_data.csv", None),
    "real_param": ("ds_real_A.csv", None),
    "real_fold": ("ds_real_A.csv", FOLD),
    "real_reset_param": ("ds_real_R.csv", None),
    "real_reset_fold": ("ds_real_R.csv", FOLD),
}


def load(config):
    path, lam_cols = CONFIGS[config]
    df = pd.read_csv(path)
    if lam_cols is not None:                       # present folded columns under the standard names
        for i, c in enumerate(lam_cols, start=1):
            df[f"noisy_l{i}"] = df[c]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    a = ap.parse_args()
    df = load(a.config)
    rows, pairs = [], []
    for s in range(N_SPLITS):
        ex.SEED = 100 + s
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=100 + s)
                      .split(df, groups=df["circuit_id"]))
        preds, npar, yte, gte = ex.run_arms(df, tr, te, feats=FEATS, arms=ARMS, fast=True,
                                            label=f"{a.config} split {s}")
        preds = {m: np.clip(p, -1.5, 1.5) for m, p in preds.items()}
        base = rmse(yte, preds["no_mitigation"])
        for m, p in preds.items():
            rows.append({"config": a.config, "split_seed": 100 + s, "model": m,
                         "rmse": rmse(yte, p),
                         "err_reduction_pct": float(100 * (1 - rmse(yte, p) / base)),
                         "qk_bandwidth": npar.get("qk_ridge_quantum_bandwidth", np.nan)
                         if m == "qk_ridge_quantum" else np.nan,
                         "n_test_circuits": int(len(np.unique(gte)))})
        for m in preds:
            for ref in REFS:
                if m == ref:
                    continue
                r = paired_diff(yte, preds[m], yte, preds[ref], gte, n_boot=N_BOOT, seed=1000 + s)
                pairs.append({"config": a.config, "split_seed": 100 + s, "model": m, "ref": ref,
                              "rmse_diff": r["diff"], "diff_lo": r["diff_lo"], "diff_hi": r["diff_hi"],
                              "model_better_resolved": bool(r["diff_hi"] < 0),
                              "model_worse_resolved": bool(r["diff_lo"] > 0)})
        pd.DataFrame(rows).to_csv(f"robust_splits_{a.config}.csv", index=False)
        pd.DataFrame(pairs).to_csv(f"robust_pairs_{a.config}.csv", index=False)
        print(f"split {s} done", flush=True)
    r = pd.DataFrame(rows)
    print(r.groupby("model")["rmse"].agg(["mean", "std", "min", "max"]).round(4).sort_values("mean").to_string())


if __name__ == "__main__":
    main()
