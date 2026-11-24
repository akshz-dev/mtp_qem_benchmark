"""The seven extended mitigation experiments for the thesis results chapter.

E1 noise-strength sweep      E5 learning curves
E2 shot-budget sweep         E6 feature ablation
E3 qubit-count scaling       E7 cross-family transfer
E4 bounded (non-periodic) encoding VQC -- tested as an arm in every experiment

Reuses the model implementations in bench.py so the main-table results and the
sweep results come from identical code paths.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import bench
from bench import (FEATS, boot_rmse_ci, dr_predict, fit_dr_vqc, qk_states,
                   quantum_gram, rmse, zne_exp, zne_poly)

SEED = 7
LAM_FEATS = ["noisy_l1", "noisy_l2", "noisy_l3"]

# ---------------------------------------------------------------- E4: bounded VQC
# The data-reuploading VQC of bench.py encodes features as raw rotation angles, so a
# feature value outside the training range wraps around the circle. Squashing the
# pre-activation through tanh confines the angle to [-pi/2, pi/2], where sin is
# monotone, so out-of-range inputs saturate instead of wrapping. E4 tests whether
# that is really the cause of the depth-extrapolation collapse.
BD_WIRES, BD_LAYERS = 4, 4
bd_dev = qml.device("default.qubit", wires=BD_WIRES)


@qml.qnode(bd_dev, interface="autograd")
def bd_circuit(x, w, b):
    for l in range(BD_LAYERS):
        for q in range(BD_WIRES):
            qml.RY((np.pi / 2) * pnp.tanh(pnp.dot(x, w[l, q]) + b[l, q]), wires=q)
            qml.RZ(b[l, q], wires=q)
        for q in range(BD_WIRES):
            qml.CNOT(wires=[q, (q + 1) % BD_WIRES])
    return qml.expval(qml.PauliZ(0))


def bd_pred(X, par):
    return par["s"][0] * bd_circuit(np.asarray(X, float), par["w"], par["b"]) + par["s"][1]


def fit_bd_vqc(Xtr, ytr, epochs=40, batch=96, lr=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    nf = Xtr.shape[1]
    par = {"w": pnp.array(rng.normal(0, 0.4, (BD_LAYERS, BD_WIRES, nf)), requires_grad=True),
           "b": pnp.array(rng.normal(0, 0.1, (BD_LAYERS, BD_WIRES)), requires_grad=True),
           "s": pnp.array([1.0, 0.0], requires_grad=True)}
    opt = qml.AdamOptimizer(stepsize=lr)

    def cost(w, b, s, xb, yb):
        return pnp.mean((s[0] * bd_circuit(xb, w, b) + s[1] - yb) ** 2)

    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            par["w"], par["b"], par["s"], _, _ = opt.step(
                cost, par["w"], par["b"], par["s"], Xtr[idx], ytr[idx])
    return par


_QK_CACHE = {}


def qk_states_n(X, bw):
    """Bandwidth-rescaled IQP embedding states on one wire per feature.

    bench.qk_states fixes six wires because the main benchmark always uses six
    features; the ablation experiment varies the feature count, so the device
    width must follow it. For six features this reproduces bench.qk_states exactly.
    """
    nf = X.shape[1]
    if nf not in _QK_CACHE:
        dev_n = qml.device("default.qubit", wires=nf)

        @qml.qnode(dev_n)
        def fm(x):
            qml.IQPEmbedding(x, wires=range(nf), n_repeats=2)
            return qml.state()

        _QK_CACHE[nf] = fm
    fm = _QK_CACHE[nf]
    return np.array([fm(bw * x) for x in X])


KIND = {"no_mitigation": "unmitigated", "zne_linear": "analytic ZNE",
        "zne_richardson2": "analytic ZNE", "zne_exponential": "analytic ZNE",
        "ridge": "classical ML", "hist_gbm": "classical ML", "mlp": "classical ML",
        "qk_ridge_quantum": "hybrid quantum-classical",
        "dr_vqc_quantum": "hybrid quantum-classical",
        "bounded_vqc_quantum": "hybrid quantum-classical"}
EVALS = {"no_mitigation": 1, "zne_linear": 3, "zne_richardson2": 3, "zne_exponential": 3,
         "ridge": 3, "hist_gbm": 3, "mlp": 3, "qk_ridge_quantum": 4,
         "dr_vqc_quantum": 4, "bounded_vqc_quantum": 4}
ALL_ARMS = list(KIND)


def run_arms(df, tr, te, feats=FEATS, arms=ALL_ARMS, fast=False, label=""):
    """Fit the requested arms on tr and predict te. Returns (preds, nparam)."""
    Xtr_r, Xte_r = df.loc[tr, feats].to_numpy(), df.loc[te, feats].to_numpy()
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gtr = df.loc[tr, "circuit_id"].to_numpy()
    sc = StandardScaler().fit(Xtr_r)
    Xtr, Xte = sc.transform(Xtr_r), sc.transform(Xte_r)
    have_lams = all(f in feats for f in LAM_FEATS)
    Yte_l = df.loc[te, LAM_FEATS].to_numpy()
    preds, npar = {}, {}
    cv = GroupKFold(n_splits=4)

    for a in arms:
        t0 = time.time()
        if a == "no_mitigation":
            preds[a] = df.loc[te, "noisy_l1"].to_numpy()
        elif a in ("zne_linear", "zne_richardson2", "zne_exponential"):
            if not have_lams:
                continue
            preds[a] = (zne_poly(Yte_l, 1) if a == "zne_linear" else
                        zne_poly(Yte_l, 2) if a == "zne_richardson2" else zne_exp(Yte_l))
        elif a == "ridge":
            m = RidgeCV(alphas=np.logspace(-4, 3, 30)).fit(Xtr, ytr)
            preds[a], npar[a] = m.predict(Xte), Xtr.shape[1] + 1
        elif a == "hist_gbm":
            grid = {"max_leaf_nodes": [31], "learning_rate": [0.1], "max_iter": [300]} if fast \
                else {"max_leaf_nodes": [15, 31], "learning_rate": [0.05, 0.1], "max_iter": [300]}
            m = GridSearchCV(HistGradientBoostingRegressor(random_state=SEED), grid,
                             cv=cv, n_jobs=1).fit(Xtr, ytr, groups=gtr)
            preds[a] = m.predict(Xte)
            npar[a] = sum(p.get_n_leaf_nodes() for pl in
                          m.best_estimator_._predictors for p in pl)
        elif a == "mlp":
            m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                             early_stopping=True, random_state=SEED).fit(Xtr, ytr)
            preds[a] = m.predict(Xte)
            npar[a] = sum(c.size for c in m.coefs_) + sum(i.size for i in m.intercepts_)
        elif a == "qk_ridge_quantum":
            bws = (0.05, 0.15, 0.4) if fast else bench.QK_BANDWIDTHS
            alphas = np.logspace(-6, 1, 8) if fast else np.logspace(-6, 1, 15)
            best = None
            for bw in bws:
                Ptr = qk_states_n(Xtr, bw)
                g = GridSearchCV(KernelRidge(kernel="precomputed"), {"alpha": alphas},
                                 cv=cv, n_jobs=1).fit(quantum_gram(Ptr), ytr, groups=gtr)
                if best is None or g.best_score_ > best[0]:
                    best = (g.best_score_, bw, g.best_estimator_, Ptr)
            _, bw, m, Ptr = best
            preds[a] = m.predict(quantum_gram(qk_states_n(Xte, bw), Ptr))
            npar[a] = len(ytr)
            npar[a + "_bandwidth"] = bw
        elif a == "dr_vqc_quantum":
            par, _ = fit_dr_vqc(Xtr, ytr, epochs=30 if fast else 40)
            preds[a] = np.asarray(dr_predict(Xte, par)).ravel()
            npar[a] = par["w"].size + par["b"].size + par["s"].size
        elif a == "bounded_vqc_quantum":
            par = fit_bd_vqc(Xtr, ytr, epochs=30 if fast else 40)
            preds[a] = np.asarray(bd_pred(Xte, par)).ravel()
            npar[a] = par["w"].size + par["b"].size + par["s"].size
        npar[a + "_sec"] = round(time.time() - t0, 2)
    print(f"  [{label}] fitted {len(preds)} arms", flush=True)
    return preds, npar, yte, df.loc[te, "circuit_id"].to_numpy()


def score(preds, npar, yte, gte, experiment, config, split, extra=""):
    base = rmse(yte, preds["no_mitigation"])
    out = []
    for m, p in preds.items():
        p = np.clip(p, -1.5, 1.5)
        lo, hi = boot_rmse_ci(yte, p, gte, n=400)
        out.append({"experiment": experiment, "config": config, "split": split,
                    "model": m, "kind": KIND[m], "rmse": rmse(yte, p),
                    "rmse_lo": lo, "rmse_hi": hi,
                    "mae": float(np.mean(np.abs(yte - p))),
                    "r2": float(1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)),
                    "err_reduction_pct": float(100 * (1 - rmse(yte, p) / base)),
                    "circuit_evals_per_pred": EVALS[m],
                    "n_params": int(npar.get(m, 0)), "fit_sec": npar.get(m + "_sec", 0.0),
                    "qk_bandwidth": npar.get("qk_ridge_quantum_bandwidth", np.nan)
                                    if m == "qk_ridge_quantum" else np.nan,
                    "n_test": len(yte), "extra": extra})
    return out


def split_A(df, seed=SEED, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=df["circuit_id"]))


def flush(rows):
    pd.DataFrame(rows).to_csv("sweep_results.csv", index=False)
    return rows


def main():
    rows = []
    main_df = pd.read_csv("benchmark_data.csv")

    # ---- E1 noise-strength sweep (plus the main 1x dataset) -------------------
    for mult, path in [(0.5, "ds_noise0.5x.csv"), (1.0, "benchmark_data.csv"),
                       (2.0, "ds_noise2.0x.csv"), (4.0, "ds_noise4.0x.csv")]:
        df = pd.read_csv(path)
        tr, te = split_A(df)
        pr, npr, yte, gte = run_arms(df, tr, te, fast=(mult != 1.0), label=f"E1 {mult}x")
        rows += score(pr, npr, yte, gte, "E1_noise_strength", f"{mult}x", "A_random_circuits",
                      extra=f"p2={0.02*mult:.3f}")
        flush(rows)

    # ---- E2 shot-budget sweep (resampled from the exact noisy values) --------
    rng = np.random.default_rng(SEED)
    for shots in [256, 1024, 4096, 0]:
        df = main_df.copy()
        for lam in ("1", "2", "3"):
            ex = df[f"exact_noisy_l{lam}"].to_numpy()
            if shots == 0:                      # analytic limit: no sampling noise
                df[f"noisy_l{lam}"] = ex
            else:
                p = np.clip((1.0 + ex) / 2.0, 0.0, 1.0)
                df[f"noisy_l{lam}"] = 2.0 * rng.binomial(shots, p) / shots - 1.0
        tr, te = split_A(df)
        pr, npr, yte, gte = run_arms(df, tr, te, fast=True, label=f"E2 {shots} shots")
        rows += score(pr, npr, yte, gte, "E2_shot_budget",
                      "analytic" if shots == 0 else str(shots), "A_random_circuits")
        flush(rows)

    # ---- E3 qubit-count scaling ----------------------------------------------
    for q, path in [(4, "ds_q4.csv"), (5, "benchmark_data.csv"), (6, "ds_q6.csv")]:
        df = pd.read_csv(path)
        tr, te = split_A(df)
        pr, npr, yte, gte = run_arms(df, tr, te, fast=(q != 5), label=f"E3 {q}q")
        rows += score(pr, npr, yte, gte, "E3_qubit_count", f"{q}q", "A_random_circuits")
        flush(rows)

    # ---- E4 bounded encoding on the depth-extrapolation split ----------------
    trB = main_df.index[main_df["depth"] <= 4].to_numpy()
    teB = main_df.index[main_df["depth"] >= 5].to_numpy()
    pr, npr, yte, gte = run_arms(main_df, trB, teB, label="E4 depth-extrap")
    rows += score(pr, npr, yte, gte, "E4_bounded_encoding", "train_depth<=4",
                  "B_depth_extrapolation")
    flush(rows)

    # ---- E5 learning curves ---------------------------------------------------
    tr, te = split_A(main_df)
    tr_groups = np.unique(main_df.loc[tr, "circuit_id"])
    arms5 = ["no_mitigation", "zne_richardson2", "ridge", "hist_gbm", "mlp",
             "qk_ridge_quantum", "dr_vqc_quantum", "bounded_vqc_quantum"]
    for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
        rs = np.random.default_rng(SEED)
        keep = rs.choice(tr_groups, size=max(8, int(len(tr_groups) * frac)), replace=False)
        tr_f = main_df.index[main_df["circuit_id"].isin(keep)].to_numpy()
        pr, npr, yte, gte = run_arms(main_df, tr_f, te, arms=arms5, fast=True,
                                     label=f"E5 frac={frac}")
        rows += score(pr, npr, yte, gte, "E5_learning_curve", f"{frac}",
                      "A_random_circuits", extra=f"n_train_circuits={len(keep)}")
        flush(rows)

    # ---- E6 feature ablation --------------------------------------------------
    sets = {"lambda_only": LAM_FEATS,
            "single_noisy_plus_structure": ["noisy_l1", "depth", "n_2q", "obs_weight"],
            "full": FEATS}
    for nm, fs in sets.items():
        pr, npr, yte, gte = run_arms(main_df, tr, te, feats=fs,
                                     arms=["no_mitigation", "ridge", "hist_gbm", "mlp",
                                           "qk_ridge_quantum", "bounded_vqc_quantum"],
                                     fast=True, label=f"E6 {nm}")
        rows += score(pr, npr, yte, gte, "E6_feature_ablation", nm, "A_random_circuits",
                      extra=f"n_features={len(fs)}")
        flush(rows)

    # ---- E7 cross-family transfer --------------------------------------------
    for train_fam, test_fam in [("hea", "tfim"), ("tfim", "hea")]:
        trF = main_df.index[main_df["family"] == train_fam].to_numpy()
        teF = main_df.index[main_df["family"] == test_fam].to_numpy()
        pr, npr, yte, gte = run_arms(main_df, trF, teF, fast=True,
                                     label=f"E7 {train_fam}->{test_fam}")
        rows += score(pr, npr, yte, gte, "E7_cross_family", f"{train_fam}_to_{test_fam}",
                      "C_cross_family")
        flush(rows)

    out = pd.DataFrame(rows)
    out.to_csv("sweep_results.csv", index=False)
    print(out.groupby(["experiment", "config"]).size().to_string())


if __name__ == "__main__":
    main()
