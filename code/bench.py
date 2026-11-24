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
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED = 7
LAMS = np.array([1.0, 2.0, 3.0])
FEATS = ["noisy_l1", "noisy_l2", "noisy_l3", "depth", "n_2q", "obs_weight"]
rng_global = np.random.default_rng(SEED)
qk_choice = {}


# ---------------------------------------------------------------- analytic ZNE
def zne_poly(Y, deg):
    """Least-squares polynomial extrapolation to zero noise."""
    c = np.polynomial.polynomial.polyfit(LAMS, Y.T, deg)
    return c[0]


def zne_exp(Y):
    """Fit E(l) = a*exp(-b*l) + c, fall back to linear on failure."""
    out = np.empty(len(Y))
    lin = zne_poly(Y, 1)
    for i, y in enumerate(Y):
        try:
            p, _ = curve_fit(lambda l, a, b, c: a * np.exp(-b * l) + c, LAMS, y,
                             p0=[y[0], 0.5, 0.0], maxfev=4000)
            v = p[0] + p[2]
            out[i] = v if np.isfinite(v) and abs(v) <= 1.5 else lin[i]
        except Exception:
            out[i] = lin[i]
    return out


# ------------------------------------------------- hybrid: quantum kernel ridge
QK_WIRES = 6
qk_dev = qml.device("default.qubit", wires=QK_WIRES)


QK_BANDWIDTHS = (0.05, 0.15, 0.4, 1.0)


@qml.qnode(qk_dev)
def feature_map(x):
    """IQP-style embedding; the fidelity kernel is |<psi_i|psi_j>|^2."""
    qml.IQPEmbedding(x, wires=range(QK_WIRES), n_repeats=2)
    return qml.state()


def qk_states(X, bw):
    """Bandwidth-rescaled embedding states. Bandwidth controls the kernel
    spectrum and hence the inductive bias, so it is tuned here rather than
    fixed at 1: Canatar, Peters, Pehlevan, Wild & Shaydulin, "Bandwidth
    Enables Generalization in Quantum Kernel Models", TMLR 2023,
    arXiv:2206.06686."""
    return np.array([feature_map(bw * x) for x in X])


def quantum_gram(Psi_a, Psi_b=None):
    Psi_b = Psi_a if Psi_b is None else Psi_b
    return np.abs(Psi_a @ Psi_b.conj().T) ** 2


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
    "mlp": (3, None), "qk_ridge_quantum": (4, None), "dr_vqc_quantum": (4, None),
}


def run_split(df, tr, te, name):
    Xtr_raw, Xte_raw = df.loc[tr, FEATS].to_numpy(), df.loc[te, FEATS].to_numpy()
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gtr, gte = df.loc[tr, "circuit_id"].to_numpy(), df.loc[te, "circuit_id"].to_numpy()
    Ytr_l, Yte_l = df.loc[tr, FEATS[:3]].to_numpy(), df.loc[te, FEATS[:3]].to_numpy()

    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    preds, nparam, secs = {}, {}, {}

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

    # hybrid 1: quantum fidelity kernel + classical ridge readout
    t = time.time()
    best = None
    for bw in QK_BANDWIDTHS:
        Ptr = qk_states(Xtr, bw)
        g = GridSearchCV(KernelRidge(kernel="precomputed"),
                         {"alpha": np.logspace(-6, 1, 15)},
                         cv=GroupKFold(n_splits=4), n_jobs=1
                         ).fit(quantum_gram(Ptr), ytr, groups=gtr)
        print(f"    qk bandwidth={bw} cv_r2={g.best_score_:.4f} "
              f"alpha={g.best_params_['alpha']:.2g}", flush=True)
        if best is None or g.best_score_ > best[0]:
            best = (g.best_score_, bw, g.best_estimator_, Ptr)
    _, bw_best, qkr, Ptr = best
    qk_choice[name] = {"bandwidth": bw_best, "alpha": float(qkr.alpha)}
    preds["qk_ridge_quantum"] = qkr.predict(
        quantum_gram(qk_states(Xte, bw_best), Ptr))
    nparam["qk_ridge_quantum"] = int(len(ytr))  # one dual coefficient per train point
    secs["qk_ridge_quantum"] = time.time() - t

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
            "circuit_evals_per_pred": COST[m][0],
            "n_params": nparam.get(m, 0),
            "train_sec": round(secs.get(m, 0.0), 2),
            "n_test": int(len(yte)), "n_train": int(len(ytr)),
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
    json.dump({"dr_vqc_train_curve_A": histA, "dr_vqc_train_curve_B": histB,
               "qk_selected_hyperparams": qk_choice},
              open("train_curves.json", "w"), indent=2)
    print(res[["split", "model", "rmse", "err_reduction_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
