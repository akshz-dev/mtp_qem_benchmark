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
    times (ridge, trees, neural network, quantum kernel); the variational arms
    are excluded for cost and the chapter says so.

Outputs: followup_results.csv, followup_paired.csv
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import experiments as ex
from bench import rmse
from stats_util import paired_diff

LEARNED = ["ridge", "hist_gbm", "mlp", "qk_ridge_quantum", "dr_vqc_quantum", "bounded_vqc_quantum"]
FEATSETS = {
    "full": ["noisy_l1", "noisy_l2", "noisy_l3", "depth", "n_2q", "obs_weight"],
    "noise_only": ["noisy_l1", "noisy_l2", "noisy_l3"],
    "noise_plus_obs_weight": ["noisy_l1", "noisy_l2", "noisy_l3", "obs_weight"],
}


def main():
    df = pd.read_csv("benchmark_data.csv")
    rows, pairs = [], []

    def flush():
        pd.DataFrame(rows).to_csv("followup_results.csv", index=False)
        pd.DataFrame(pairs).to_csv("followup_paired.csv", index=False)

    # ---- F1 ------------------------------------------------------------------
    trB = df.index[df["depth"] <= 4].to_numpy()
    teB = df.index[df["depth"] >= 5].to_numpy()
    for name, feats in FEATSETS.items():
        pr, npr, yte, gte = ex.run_arms(df, trB, teB, feats=feats,
                                        arms=["no_mitigation"] + LEARNED,
                                        fast=False, label=f"F1 {name}")
        rows += ex.score(pr, npr, yte, gte, "F1_ood_features", name, "B_depth_extrapolation",
                         extra=f"n_features={len(feats)}")
        for m in LEARNED:
            if m in pr:
                r = paired_diff(yte, np.clip(pr[m], -1.5, 1.5), yte, pr["no_mitigation"], gte, seed=41)
                pairs.append({"experiment": "F1_ood_features", "config": name, "model": m,
                              "comparison": "arm_minus_unmitigated", **r})
        flush()

    # ---- F2 ------------------------------------------------------------------
    tr, te = ex.split_A(df)
    tr_groups = np.unique(df.loc[tr, "circuit_id"])
    arms = ["no_mitigation", "zne_linear", "ridge", "hist_gbm", "mlp", "qk_ridge_quantum"]
    for n_circ in (31, 63):
        for draw in range(20):
            rs = np.random.default_rng(1000 * n_circ + draw)
            keep = rs.choice(tr_groups, size=n_circ, replace=False)
            tr_f = df.index[df["circuit_id"].isin(keep)].to_numpy()
            pr, npr, yte, gte = ex.run_arms(df, tr_f, te, arms=arms, fast=True,
                                            label=f"F2 n={n_circ} draw={draw}")
            base = rmse(yte, pr["no_mitigation"])
            for m, p in pr.items():
                rows.append({"experiment": "F2_learning_variance", "config": str(n_circ),
                             "split": "A_random_circuits", "model": m, "draw": draw,
                             "rmse": rmse(yte, np.clip(p, -1.5, 1.5)),
                             "err_reduction_pct": float(100 * (1 - rmse(yte, np.clip(p, -1.5, 1.5)) / base)),
                             "n_train_circuits": n_circ})
        flush()
    flush()


if __name__ == "__main__":
    main()
