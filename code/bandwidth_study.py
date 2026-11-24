"""Quantum-kernel bandwidth sensitivity (Section 5.x of the thesis).

Evaluates fidelity-kernel ridge regression on Split A at each fixed bandwidth,
with the ridge penalty still chosen by grouped cross-validation, so the reported
error isolates the effect of bandwidth alone.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

from bench import FEATS, boot_rmse_ci, qk_states, quantum_gram, rmse
from experiments import split_A

BANDWIDTHS = [0.02, 0.05, 0.15, 0.4, 1.0, 2.0]

df = pd.read_csv("benchmark_data.csv")
tr, te = split_A(df)
sc = StandardScaler().fit(df.loc[tr, FEATS].to_numpy())
Xtr = sc.transform(df.loc[tr, FEATS].to_numpy())
Xte = sc.transform(df.loc[te, FEATS].to_numpy())
ytr = df.loc[tr, "ideal"].to_numpy()
yte = df.loc[te, "ideal"].to_numpy()
gtr = df.loc[tr, "circuit_id"].to_numpy()
gte = df.loc[te, "circuit_id"].to_numpy()
base = rmse(yte, df.loc[te, "noisy_l1"].to_numpy())

rows = []
for bw in BANDWIDTHS:
    Ptr, Pte = qk_states(Xtr, bw), qk_states(Xte, bw)
    Ktr = quantum_gram(Ptr)
    off = Ktr[~np.eye(len(Ktr), dtype=bool)]
    g = GridSearchCV(KernelRidge(kernel="precomputed"),
                     {"alpha": np.logspace(-6, 1, 15)},
                     cv=GroupKFold(n_splits=4), n_jobs=1).fit(Ktr, ytr, groups=gtr)
    pred = np.clip(g.predict(quantum_gram(Pte, Ptr)), -1.5, 1.5)
    lo, hi = boot_rmse_ci(yte, pred, gte, n=400)
    rows.append({"bandwidth": bw, "alpha_selected": float(g.best_estimator_.alpha),
                 "cv_r2": float(g.best_score_), "rmse": rmse(yte, pred),
                 "rmse_lo": lo, "rmse_hi": hi,
                 "err_reduction_pct": float(100 * (1 - rmse(yte, pred) / base)),
                 "mean_offdiag_kernel": float(off.mean()),
                 "std_offdiag_kernel": float(off.std()),
                 "unmitigated_rmse": base})
    print(f"  bw={bw}: rmse={rows[-1]['rmse']:.4f} "
          f"mean_offdiag={rows[-1]['mean_offdiag_kernel']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv("bandwidth_study.csv", index=False)

print(pd.DataFrame(rows)[["bandwidth", "rmse", "err_reduction_pct",
                          "mean_offdiag_kernel"]].round(4).to_string(index=False))
