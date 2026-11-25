"""Benchmark mitigation models: analytic ZNE, classical ML, hybrid quantum-classical.

Splits are grouped by circuit_id so no circuit appears in both train and test.
  split A: random circuit hold-out (70/30)
  split B: depth extrapolation (train depth<=4, test depth>=5)
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from kernel_models import KERNEL_MODELS, fit_kernel_ridge, qk_states, quantum_gram
from stats_util import paired_diff

SEED = 7
LAMS = np.array([1.0, 2.0, 3.0])
FEATS = ["noisy_l1", "noisy_l2", "noisy_l3", "depth", "n_2q", "obs_weight"]
rng_global = np.random.default_rng(SEED)
kernel_choices = {name: {} for name in KERNEL_MODELS}


# ---------------------------------------------------------------- analytic ZNE
def zne_poly(Y, deg, lams=LAMS):
    """Extrapolate to zero with a shared grid or one grid per observation."""
    Y, lams = np.asarray(Y), np.asarray(lams)
    if lams.ndim == 1:
        return np.polynomial.polynomial.polyfit(lams, Y.T, deg)[0]
    if lams.shape != Y.shape or Y.ndim != 2:
        raise ValueError("Expected a shared ZNE grid or one grid per observation")

    # Circuits with the same executed fold ratios can share a polynomial fit.
    grids, grid_index = np.unique(lams, axis=0, return_inverse=True)
    zero_noise = np.empty(len(Y))
    for i, grid in enumerate(grids):
        rows = grid_index == i
        zero_noise[rows] = np.polynomial.polynomial.polyfit(grid, Y[rows].T, deg)[0]
    return zero_noise


def zne_exp(Y, lams=LAMS):
    """Fit an exponential per observation, with a linear fallback on the same grid."""
    out = np.empty(len(Y))
    lin = zne_poly(Y, 1, lams)
    for i, (grid, y) in enumerate(zip(np.broadcast_to(lams, np.shape(Y)), Y)):
        try:
            p, _ = curve_fit(lambda l, a, b, c: a * np.exp(-b * l) + c, grid, y,
                             p0=[y[0], 0.5, 0.0], maxfev=4000)
            v = p[0] + p[2]
            out[i] = v if np.isfinite(v) and abs(v) <= 1.5 else lin[i]
        except Exception:
            out[i] = lin[i]
    return out


# --------------------------------------- hybrid: data-reuploading VQC regressor
DR_WIRES, DR_LAYERS = 4, 4
dr_dev = qml.device("default.qubit", wires=DR_WIRES)


@qml.qnode(dr_dev, interface="autograd")
def dr_circuit(x, w, b):
    for l in range(DR_LAYERS):
        for q in range(DR_WIRES):
            qml.RY(pnp.dot(x, w[l, q]) + b[l, q], wires=q)
            qml.RZ(b[l, q], wires=q)
        for q in range(DR_WIRES):
            qml.CNOT(wires=[q, (q + 1) % DR_WIRES])
    return qml.expval(qml.PauliZ(0))


def dr_predict(X, par):
    w, b, s = par["w"], par["b"], par["s"]
    return s[0] * dr_circuit(np.asarray(X, dtype=float), w, b) + s[1]


def fit_dr_vqc(Xtr, ytr, epochs=40, batch=96, lr=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    nf = Xtr.shape[1]
    par = {
        "w": pnp.array(rng.normal(0, 0.4, (DR_LAYERS, DR_WIRES, nf)), requires_grad=True),
        "b": pnp.array(rng.normal(0, 0.1, (DR_LAYERS, DR_WIRES)), requires_grad=True),
        "s": pnp.array([1.0, 0.0], requires_grad=True),
    }
    opt = qml.AdamOptimizer(stepsize=lr)

    def cost(w, b, s, xb, yb):
        pred = s[0] * dr_circuit(xb, w, b) + s[1]
        return pnp.mean((pred - yb) ** 2)

    n = len(Xtr)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], ytr[idx]
            par["w"], par["b"], par["s"], _, _ = opt.step(
                cost, par["w"], par["b"], par["s"], xb, yb)
        if ep % 8 == 0 or ep == epochs - 1:
            hist.append((ep, float(cost(par["w"], par["b"], par["s"],
                                        Xtr, ytr))))
            print(f"    dr-vqc epoch {ep} mse={hist[-1][1]:.5f}", flush=True)
    return par, hist


# --------------------------------------------------------------------- metrics
def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def boot_rmse_ci(y, p, groups, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in uq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([gi[g] for g in pick])
        vals.append(rmse(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# translates to circuit evaluations needed per test prediction, and free params
COST = {
    "no_mitigation":   (1, 0), "zne_linear": (3, 0), "zne_richardson2": (3, 0),
    "zne_exponential": (3, 0), "ridge": (3, None), "hist_gbm": (3, None),
    "mlp": (3, None), "rbf_krr": (3, None), "dr_vqc_quantum": (4, None),
}


def run_split(df, tr, te, name):
    Xtr_raw, Xte_raw = df.loc[tr, FEATS].to_numpy(), df.loc[te, FEATS].to_numpy()
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gtr, gte = df.loc[tr, "circuit_id"].to_numpy(), df.loc[te, "circuit_id"].to_numpy()
    Yte_l = df.loc[te, FEATS[:3]].to_numpy()

    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    preds, nparam, secs, tuning = {}, {}, {}, {}

    preds["no_mitigation"] = Yte_l[:, 0]
    preds["zne_linear"] = zne_poly(Yte_l, 1)
    preds["zne_richardson2"] = zne_poly(Yte_l, 2)
    preds["zne_exponential"] = zne_exp(Yte_l)

    t = time.time()
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 30)).fit(Xtr, ytr)
    preds["ridge"] = ridge.predict(Xte)
    nparam["ridge"] = int(Xtr.shape[1] + 1)
    secs["ridge"] = time.time() - t

    t = time.time()
    gbm = GridSearchCV(HistGradientBoostingRegressor(random_state=SEED),
                       {"max_leaf_nodes": [15, 31], "learning_rate": [0.05, 0.1],
                        "max_iter": [300]},
                       cv=GroupKFold(n_splits=4), n_jobs=1
                       ).fit(Xtr, ytr, groups=gtr)
    preds["hist_gbm"] = gbm.predict(Xte)
    nparam["hist_gbm"] = int(sum(p.get_n_leaf_nodes() for pl in
                                 gbm.best_estimator_._predictors for p in pl))
    secs["hist_gbm"] = time.time() - t

    t = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, early_stopping=True,
                       random_state=SEED).fit(Xtr, ytr)
    preds["mlp"] = mlp.predict(Xte)
    nparam["mlp"] = int(sum(c.size for c in mlp.coefs_) + sum(i.size for i in mlp.intercepts_))
    secs["mlp"] = time.time() - t

    for arm in KERNEL_MODELS:
        t = time.time()
        model = fit_kernel_ridge(arm, Xtr_raw, ytr, gtr)
        scale_name = "bandwidth" if arm == "qk_ridge_quantum" else "gamma"
        tuning[arm] = {
            "kernel_scale": model.scale, "kernel_alpha": model.alpha,
            "kernel_cv_r2": model.cv_r2, "kernel_candidates": model.n_candidates,
        }
        kernel_choices[arm][name] = {scale_name: model.scale, "alpha": model.alpha,
                                     "cv_r2": model.cv_r2, "n_candidates": model.n_candidates}
        preds[arm] = model.predict(Xte_raw)
        nparam[arm] = len(ytr)
        secs[arm] = time.time() - t

    # hybrid 2: data-reuploading variational circuit + classical affine readout
    t = time.time()
    par, hist = fit_dr_vqc(Xtr, ytr)
    preds["dr_vqc_quantum"] = np.array(dr_predict(Xte, par)).ravel()
    nparam["dr_vqc_quantum"] = int(par["w"].size + par["b"].size + par["s"].size)
    secs["dr_vqc_quantum"] = time.time() - t

    base = rmse(yte, preds["no_mitigation"])
    rows = []
    for m, p in preds.items():
        p = np.clip(p, -1.5, 1.5)
        lo, hi = boot_rmse_ci(yte, p, gte)
        rows.append({
            "split": name, "model": m,
            "kind": ("quantum" in m and "hybrid quantum-classical")
                    or ("zne" in m and "analytic ZNE")
                    or (m == "no_mitigation" and "unmitigated") or "classical ML",
            "rmse": rmse(yte, p), "rmse_lo": lo, "rmse_hi": hi,
            "mae": float(np.mean(np.abs(yte - p))),
            "median_abs_err": float(np.median(np.abs(yte - p))),
            "max_abs_err": float(np.max(np.abs(yte - p))),
            "r2": float(1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)),
            "err_reduction_pct": float(100 * (1 - rmse(yte, p) / base)),
            "circuit_evals_per_pred": 3 + len(ytr) if m == "qk_ridge_quantum" else COST[m][0],
            "n_params": nparam.get(m, 0),
            "train_sec": round(secs.get(m, 0.0), 2),
            "n_test": int(len(yte)), "n_train": int(len(ytr)),
            **tuning.get(m, {}),
        })
    return pd.DataFrame(rows), preds, yte, df.loc[te].reset_index(drop=True), hist


def main():
    df = pd.read_csv("benchmark_data.csv")
    out, pred_store = [], {}

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["circuit_id"]))
    print("split A: random circuit hold-out", flush=True)
    rA, pA, yA, dA, histA = run_split(df, tr, te, "A_random_circuits")
    out.append(rA)
    pred_store["A"] = (pA, yA, dA)

    trB = df.index[df["depth"] <= 4].to_numpy()
    teB = df.index[df["depth"] >= 5].to_numpy()
    print("split B: depth extrapolation (train<=4, test>=5)", flush=True)
    rB, pB, yB, dB, histB = run_split(df, trB, teB, "B_depth_extrapolation")
    out.append(rB)
    pred_store["B"] = (pB, yB, dB)

    res = pd.concat(out, ignore_index=True)
    res.to_csv("benchmark_results.csv", index=False)

    # per-observable and per-depth breakdown on split A
    pA, yA, dA = pred_store["A"]
    br = []
    for m, p in pA.items():
        p = np.clip(p, -1.5, 1.5)
        for key in ["observable", "depth", "family"]:
            for v, g in dA.groupby(key).groups.items():
                g = np.asarray(g)
                br.append({"model": m, "facet": key, "level": str(v),
                           "rmse": rmse(yA[g], p[g]), "n": len(g)})
    pd.DataFrame(br).to_csv("breakdown_splitA.csv", index=False)

    np.savez("predictions_splitA.npz", y=yA,
             **{m: np.clip(p, -1.5, 1.5) for m, p in pA.items()})
    np.savez("predictions_splitB.npz", y=yB,
             **{m: np.clip(p, -1.5, 1.5) for m, p in pB.items()})
    contrasts = []
    for split, (preds, y, test) in (
        ("A_random_circuits", (pA, yA, dA)),
        ("B_depth_extrapolation", (pB, yB, dB)),
    ):
        contrasts.append({"split": split, **matched_kernel_difference(preds, y, test["circuit_id"])})
    pd.DataFrame(contrasts).to_csv("paired_kernel_main.csv", index=False)
    json.dump({"dr_vqc_train_curve_A": histA, "dr_vqc_train_curve_B": histB,
               "qk_selected_hyperparams": kernel_choices["qk_ridge_quantum"],
               "rbf_selected_hyperparams": kernel_choices["rbf_krr"]},
              open("train_curves.json", "w"), indent=2)
    print(res[["split", "model", "rmse", "err_reduction_pct"]].to_string(index=False))


def matched_kernel_difference(preds, y, groups):
    """Paired circuit-bootstrap RMSE(RBF) minus RMSE(fidelity kernel)."""
    rbf = np.clip(preds["rbf_krr"], -1.5, 1.5)
    qk = np.clip(preds["qk_ridge_quantum"], -1.5, 1.5)
    return paired_diff(y, rbf, y, qk, groups, seed=SEED)


if __name__ == "__main__":
    main()
