"""Quantum-kernel bandwidth sensitivity (Chapter 5).

Fix the IQP bandwidth in each sweep arm. Select the ridge penalty on four
circuit-grouped inner folds with fold-local feature scaling.
"""

import numpy as np
import pandas as pd

from bench import FEATS, boot_rmse_ci, rmse
from experiments import split_A
from kernel_models import fit_kernel_ridge, quantum_gram

BANDWIDTHS = (0.02, 0.05, 0.15, 0.4, 1.0, 2.0)


def main():
    df = pd.read_csv("benchmark_data.csv")
    tr, te = split_A(df)
    Xtr, Xte = df.loc[tr, FEATS].to_numpy(), df.loc[te, FEATS].to_numpy()
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gtr, gte = df.loc[tr, "circuit_id"].to_numpy(), df.loc[te, "circuit_id"].to_numpy()
    base = rmse(yte, df.loc[te, "noisy_l1"].to_numpy())

    rows = []
    for bw in BANDWIDTHS:
        model = fit_kernel_ridge("qk_ridge_quantum", Xtr, ytr, gtr, fixed_scale=bw)
        gram = quantum_gram(model.training_representation)
        off_diagonal = gram[~np.eye(len(gram), dtype=bool)]
        pred = np.clip(model.predict(Xte), -1.5, 1.5)
        error = rmse(yte, pred)
        lo, hi = boot_rmse_ci(yte, pred, gte, n=400)
        rows.append({"bandwidth": bw, "alpha_selected": model.alpha,
                     "cv_r2": model.cv_r2, "rmse": error,
                     "rmse_lo": lo, "rmse_hi": hi,
                     "err_reduction_pct": float(100 * (1 - error / base)),
                     "mean_offdiag_kernel": float(off_diagonal.mean()),
                     "std_offdiag_kernel": float(off_diagonal.std()),
                     "unmitigated_rmse": base})
        print(f"  bw={bw}: rmse={error:.4f} mean_offdiag={off_diagonal.mean():.4f}", flush=True)
        pd.DataFrame(rows).to_csv("bandwidth_study.csv", index=False)


if __name__ == "__main__":
    main()
