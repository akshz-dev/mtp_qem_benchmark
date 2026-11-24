"""Sensitivity of the two variational arms to their initialisation seed.

The main benchmark fits each variational circuit once (seed 7, 40 epochs).
Re-implementations of the same arm gave very different test errors on the same
split (0.0557 vs about 0.18), so the single fit cannot be reported without its
spread. Here each arm is refitted with ten initialisation / minibatch seeds on
the idealized Split A and Split B, at the 40-epoch budget of the main run, and
the data-reuploading arm also at the 30-epoch budget used by the sweeps.
Data, split, feature scaling and every other setting are those of bench.py.
Output: vqc_seed_sweep.csv
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import experiments as ex
from bench import FEATS, dr_predict, fit_dr_vqc, rmse, boot_rmse_ci

df = pd.read_csv("benchmark_data.csv")
trA, teA = ex.split_A(df)
trB = df.index[df["depth"] <= 4].to_numpy(); teB = df.index[df["depth"] >= 5].to_numpy()
rows = []
for split, (tr, te) in [("A_random_circuits", (trA, teA)), ("B_depth_extrapolation", (trB, teB))]:
    sc = StandardScaler().fit(df.loc[tr, FEATS].to_numpy())
    Xtr, Xte = sc.transform(df.loc[tr, FEATS].to_numpy()), sc.transform(df.loc[te, FEATS].to_numpy())
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gte = df.loc[te, "circuit_id"].to_numpy()
    base = rmse(yte, df.loc[te, "noisy_l1"].to_numpy())
    jobs = [("dr_vqc_quantum", 40), ("bounded_vqc_quantum", 40)] + ([("dr_vqc_quantum", 30)] if split.startswith("A") else [])
    for arm, ep in jobs:
        for seed in range(10):
            t0 = time.time()
            if arm == "dr_vqc_quantum":
                par, hist = fit_dr_vqc(Xtr, ytr, epochs=ep, seed=seed)
                p = np.asarray(dr_predict(Xte, par)).ravel()
                train_mse = hist[-1][1]
            else:
                par = ex.fit_bd_vqc(Xtr, ytr, epochs=ep, seed=seed)
                p = np.asarray(ex.bd_pred(Xte, par)).ravel()
                train_mse = float(np.mean((np.asarray(ex.bd_pred(Xtr, par)).ravel() - ytr) ** 2))
            p = np.clip(p, -1.5, 1.5)
            lo, hi = boot_rmse_ci(yte, p, gte, n=300)
            rows.append({"split": split, "model": arm, "epochs": ep, "seed": seed,
                         "rmse": rmse(yte, p), "rmse_lo": lo, "rmse_hi": hi,
                         "train_mse": train_mse,
                         "err_reduction_pct": 100 * (1 - rmse(yte, p) / base),
                         "fit_sec": round(time.time() - t0, 1)})
            pd.DataFrame(rows).to_csv("vqc_seed_sweep.csv", index=False)
            print(f"{split} {arm} ep={ep} seed={seed} rmse={rows[-1]['rmse']:.4f}", flush=True)
