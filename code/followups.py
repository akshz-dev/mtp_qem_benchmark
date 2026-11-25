"""Follow-up analyses (idealized regime).

F1  Out-of-distribution feature dependence. On the depth-extrapolation split
    (train depth <= 4, test depth >= 5), refit every learned arm with three
    feature sets: full (three noise-scaled values + depth, n_2q, obs_weight),
    noise-scaled values only, and noise-scaled values + obs_weight. depth and
    n_2q are the only features whose test values lie outside the training range,
    so this separates "learned models cannot extrapolate" from "learned models
    are given inputs that leave the training support".
F2  Learning-curve variance. The single-draw learning curve (E5) reports one
    random subset per training size. Here 20 independent circuit subsets are
    drawn at 31 and 63 training circuits for the arms cheap enough to refit 20
    times (ridge, trees, neural network, and both kernels); the variational arms
    are excluded for cost and the chapter says so.

Outputs: followup_results.csv, followup_paired.csv, paired_kernel_followups.csv
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import experiments as ex
from bench import KERNEL_MODELS, matched_kernel_difference, rmse
from stats_util import paired_diff

LEARNED = ("ridge", "hist_gbm", "mlp", *KERNEL_MODELS, "dr_vqc_quantum", "bounded_vqc_quantum")
FEATSETS = {
    "full": ["noisy_l1", "noisy_l2", "noisy_l3", "depth", "n_2q", "obs_weight"],
    "noise_only": ["noisy_l1", "noisy_l2", "noisy_l3"],
    "noise_plus_obs_weight": ["noisy_l1", "noisy_l2", "noisy_l3", "obs_weight"],
}


def followup_cases():
    df = pd.read_csv("benchmark_data.csv")
    trB = df.index[df["depth"] <= 4].to_numpy()
    teB = df.index[df["depth"] >= 5].to_numpy()
    for name, feats in FEATSETS.items():
        yield (ex.SweepCase(df, trB, teB, "F1_ood_features", name, "B_depth_extrapolation",
                            feats=tuple(feats), arms=("no_mitigation", *LEARNED),
                            extra=f"n_features={len(feats)}"), None)

    tr, te = ex.split_A(df)
    tr_groups = np.unique(df.loc[tr, "circuit_id"])
    arms = ("no_mitigation", "zne_linear", "ridge", "hist_gbm", "mlp", *KERNEL_MODELS)
    for n_circ in (31, 63):
        for draw in range(20):
            rs = np.random.default_rng(1000 * n_circ + draw)
            keep = rs.choice(tr_groups, size=n_circ, replace=False)
            tr_f = df.index[df["circuit_id"].isin(keep)].to_numpy()
            yield (ex.SweepCase(df, tr_f, te, "F2_learning_variance", str(n_circ),
                                fast=True, arms=arms), draw)


def run_followup_case(case, draw):
    pr, npar, yte, gte = ex.run_arms(case.df, case.train, case.test, feats=list(case.feats),
                                     arms=case.arms, fast=case.fast,
                                     label=f"{case.experiment} {case.config} draw={draw}")
    kernel_pair = {"experiment": case.experiment, "config": case.config,
                   "split": case.split, "draw": draw,
                   **matched_kernel_difference(pr, yte, gte)}
    pairs = []
    if case.experiment == "F1_ood_features":
        rows = ex.score(pr, npar, yte, gte, case.experiment, case.config,
                        case.split, extra=case.extra)
        for model in LEARNED:
            if model in pr:
                r = paired_diff(yte, np.clip(pr[model], -1.5, 1.5), yte,
                                pr["no_mitigation"], gte, seed=41)
                pairs.append({"experiment": case.experiment, "config": case.config,
                              "model": model, "comparison": "arm_minus_unmitigated", **r})
    else:
        base = rmse(yte, pr["no_mitigation"])
        rows = []
        for model, prediction in pr.items():
            error = rmse(yte, np.clip(prediction, -1.5, 1.5))
            rows.append({"experiment": case.experiment, "config": case.config,
                         "split": case.split, "model": model, "draw": draw,
                         "rmse": error, "err_reduction_pct": 100 * (1 - error / base),
                         "n_train_circuits": int(case.config)})
    return rows, pairs, kernel_pair


def main():
    rows, pairs, kernel_pairs = [], [], []
    for case, draw in followup_cases():
        records, comparisons, kernel_pair = run_followup_case(case, draw)
        rows.extend(records)
        pairs.extend(comparisons)
        kernel_pairs.append(kernel_pair)
        pd.DataFrame(rows).to_csv("followup_results.csv", index=False)
        pd.DataFrame(pairs).to_csv("followup_paired.csv", index=False)
        pd.DataFrame(kernel_pairs).to_csv("paired_kernel_followups.csv", index=False)


if __name__ == "__main__":
    main()
