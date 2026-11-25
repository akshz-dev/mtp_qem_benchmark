"""Robustness of the headline comparisons to the train/test split.

Every result in the main benchmark uses one circuit-grouped 70/30 split (seed 7).
This script repeats the comparison over N_SPLITS circuit-grouped holdouts
of the same data and bootstraps paired differences on shared test circuits.

Usage: python robustness.py --config all --workers 4

What varies per repetition: the train/test split, the cross-validation folds
that follow from it, and the scikit-learn model seeds (MLP initialisation,
tree subsampling). What does not vary: the dataset, and the variational-circuit
initialisation seed (bound at definition time in bench.py / experiments.py).
Outputs: robust_splits_<config>.csv, robust_pairs_<config>.csv and, with
--config all, robustness_summary.csv.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

import experiments as ex
from bench import FEATS, rmse
from experiments_realistic import FOLD_FEATS, as_param
from stats_util import paired_diff

N_SPLITS = 10
N_BOOT = 1000
ARMS = ["no_mitigation", "zne_linear", "zne_richardson2", "zne_exponential",
        "ridge", "hist_gbm", "mlp", "qk_ridge_quantum", "rbf_krr", "dr_vqc_quantum",
        "bounded_vqc_quantum"]
REFERENCE_NAMES = {"no_mitigation": "unmit", "zne_linear": "zne_lin",
                   "mlp": "mlp", "rbf_krr": "rbf"}
CONFIGS = {
    "ideal": ("benchmark_data.csv", False),
    "real_param": ("ds_real_A.csv", False),
    "real_fold": ("ds_real_A.csv", True),
    "real_reset_param": ("ds_real_R.csv", False),
    "real_reset_fold": ("ds_real_R.csv", True),
}


def load(config):
    path, folded = CONFIGS[config]
    df = pd.read_csv(path)
    return as_param(df, FOLD_FEATS) if folded else df


def run_split(job):
    config, seed = job
    df = load(config)
    ex.SEED = seed
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
                  .split(df, groups=df["circuit_id"]))
    with threadpool_limits(limits=1):
        preds, npar, yte, gte = ex.run_arms(df, tr, te, feats=FEATS, arms=ARMS, fast=True,
                                           label=f"{config} split {seed}")
    preds = {m: np.clip(p, -1.5, 1.5) for m, p in preds.items()}
    base = rmse(yte, preds["no_mitigation"])
    rows, pairs = [], []
    for m, p in preds.items():
        rows.append({"config": config, "split_seed": seed, "model": m,
                     "rmse": rmse(yte, p),
                     "err_reduction_pct": 100 * (1 - rmse(yte, p) / base),
                     "qk_bandwidth": npar.get(m + "_scale", np.nan) if m == "qk_ridge_quantum" else np.nan,
                     "rbf_gamma": npar.get(m + "_scale", np.nan) if m == "rbf_krr" else np.nan,
                     "kernel_alpha": npar.get(m + "_alpha", np.nan),
                     "kernel_candidates": npar.get(m + "_candidates", np.nan),
                     "n_test_circuits": len(np.unique(gte))})
        for ref in REFERENCE_NAMES:
            if m == ref:
                continue
            r = paired_diff(yte, p, yte, preds[ref], gte, n_boot=N_BOOT, seed=900 + seed)
            pairs.append({"config": config, "split_seed": seed, "model": m, "ref": ref,
                          "rmse_diff": r["diff"], "diff_lo": r["diff_lo"], "diff_hi": r["diff_hi"],
                          "model_better_resolved": bool(r["diff_hi"] < 0),
                          "model_worse_resolved": bool(r["diff_lo"] > 0)})
    print(f"{config} seed {seed} complete", flush=True)
    return rows, pairs


def summarize(rows, pairs):
    ranked = rows.assign(rank=rows.groupby(["config", "split_seed"])["rmse"].rank())
    output = []
    for (config, model), arm in ranked.groupby(["config", "model"]):
        row = dict(config=config, model=model, mean=arm.rmse.mean(), sd=arm.rmse.std(),
                   red_pct=arm.err_reduction_pct.mean(), mean_rank=arm["rank"].mean())
        for ref, name in REFERENCE_NAMES.items():
            sample = pairs[(pairs.config == config) & (pairs.model == model) & (pairs.ref == ref)]
            for direction in ("better", "worse"):
                count = sample[f"model_{direction}_resolved"].sum()
                row[f"{direction}_than_{name}"] = f"{count}/{len(sample)}" if len(sample) else "-"
        output.append(row)
    return pd.DataFrame(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["all", *CONFIGS])
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.workers < 1:
        ap.error("--workers must be positive")
    configs = list(CONFIGS) if a.config == "all" else [a.config]
    jobs = [(config, seed) for config in configs for seed in range(100, 100 + N_SPLITS)]
    if a.workers == 1:
        completed = list(map(run_split, jobs))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            completed = list(pool.map(run_split, jobs))
    rows = pd.DataFrame(row for group, _ in completed for row in group)
    pairs = pd.DataFrame(row for _, group in completed for row in group)
    for config in configs:
        rows[rows.config == config].to_csv(f"robust_splits_{config}.csv", index=False)
        pairs[pairs.config == config].to_csv(f"robust_pairs_{config}.csv", index=False)
    summary = summarize(rows, pairs)
    if a.config == "all":
        summary.to_csv("robustness_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
