"""The seven extended mitigation experiments for the thesis results chapter.

E1 noise-strength sweep      E5 learning curves
E2 shot-budget sweep         E6 feature ablation
E3 qubit-count scaling       E7 cross-family transfer
E4 bounded (non-periodic) encoding VQC -- tested as an arm in every experiment

Reuses the model implementations in bench.py so the main-table results and the
sweep results come from identical code paths.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from bench import (FEATS, KERNEL_MODELS, LAMS, boot_rmse_ci, dr_predict, fit_dr_vqc,
                   matched_kernel_difference, rmse, zne_exp, zne_poly)
from kernel_models import fit_kernel_ridge, qk_states as qk_states_n, quantum_gram
from noise_sweep_data import GENERATOR, NOISE_LEVELS

SEED = 7
LAM_FEATS = ["noisy_l1", "noisy_l2", "noisy_l3"]
ZNE_SCALE_COLUMNS = ["zne_scale_l1", "zne_scale_l2", "zne_scale_l3"]
NOISE_EXPERIMENT = "E1_noise_strength"

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


KIND = {"no_mitigation": "unmitigated", "zne_linear": "analytic ZNE",
        "zne_richardson2": "analytic ZNE", "zne_exponential": "analytic ZNE",
        "ridge": "classical ML", "hist_gbm": "classical ML", "mlp": "classical ML",
        "rbf_krr": "classical ML",
        "qk_ridge_quantum": "hybrid quantum-classical",
        "dr_vqc_quantum": "hybrid quantum-classical",
        "bounded_vqc_quantum": "hybrid quantum-classical"}
EVALS = {"no_mitigation": 1, "zne_linear": 3, "zne_richardson2": 3, "zne_exponential": 3,
         "ridge": 3, "hist_gbm": 3, "mlp": 3, "rbf_krr": 3,
         "dr_vqc_quantum": 4, "bounded_vqc_quantum": 4}
ALL_ARMS = list(KIND)


def run_arms(df, tr, te, feats=FEATS, arms=ALL_ARMS, fast=False, label=""):
    """Return outer-test predictions, fit details, targets, and circuit groups."""
    Xtr_r, Xte_r = df.loc[tr, feats].to_numpy(), df.loc[te, feats].to_numpy()
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gtr = df.loc[tr, "circuit_id"].to_numpy()
    sc = StandardScaler().fit(Xtr_r)
    Xtr, Xte = sc.transform(Xtr_r), sc.transform(Xte_r)
    have_lams = all(f in feats for f in LAM_FEATS)
    Yte_l = df.loc[te, LAM_FEATS].to_numpy()
    zne_scales = LAMS
    if have_lams and any(column in df for column in ZNE_SCALE_COLUMNS):
        zne_scales = df.loc[te, ZNE_SCALE_COLUMNS].to_numpy()
    preds, npar = {}, {}
    cv = GroupKFold(n_splits=4)

    for a in arms:
        t0 = time.time()
        if a == "no_mitigation":
            preds[a] = df.loc[te, "noisy_l1"].to_numpy()
        elif a in ("zne_linear", "zne_richardson2", "zne_exponential"):
            if not have_lams:
                continue
            preds[a] = (zne_poly(Yte_l, 1, zne_scales) if a == "zne_linear" else
                        zne_poly(Yte_l, 2, zne_scales) if a == "zne_richardson2"
                        else zne_exp(Yte_l, zne_scales))
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
        elif a in KERNEL_MODELS:
            m = fit_kernel_ridge(a, Xtr_r, ytr, gtr, fast=fast)
            preds[a] = m.predict(Xte_r)
            npar[a] = len(ytr)
            npar[a + "_scale"] = m.scale
            npar[a + "_alpha"] = m.alpha
            npar[a + "_cv_r2"] = m.cv_r2
            npar[a + "_candidates"] = m.n_candidates
            if a == "qk_ridge_quantum":
                npar[a + "_bandwidth"] = m.scale
        elif a == "dr_vqc_quantum":
            par, _ = fit_dr_vqc(Xtr, ytr, epochs=30 if fast else 40)
            preds[a] = np.asarray(dr_predict(Xte, par)).ravel()
            npar[a] = par["w"].size + par["b"].size + par["s"].size
        elif a == "bounded_vqc_quantum":
            par = fit_bd_vqc(Xtr, ytr, epochs=30 if fast else 40)
            preds[a] = np.asarray(bd_pred(Xte, par)).ravel()
            npar[a] = par["w"].size + par["b"].size + par["s"].size
        npar[a + "_sec"] = round(time.time() - t0, 2)
    npar["measurement_inputs"] = sum(f in feats for f in LAM_FEATS)
    print(f"  [{label}] fitted {len(preds)} arms", flush=True)
    return preds, npar, yte, df.loc[te, "circuit_id"].to_numpy()


def score(preds, npar, yte, gte, experiment, config, split, extra=""):
    base = rmse(yte, preds["no_mitigation"])
    out = []
    for m, p in preds.items():
        p = np.clip(p, -1.5, 1.5)
        lo, hi = boot_rmse_ci(yte, p, gte, n=400)
        circuits = EVALS.get(m)
        if m in KERNEL_MODELS:
            circuits = npar["measurement_inputs"]
            if m == "qk_ridge_quantum":
                circuits += npar[m]
        out.append({"experiment": experiment, "config": config, "split": split,
                    "model": m, "kind": KIND[m], "rmse": rmse(yte, p),
                    "rmse_lo": lo, "rmse_hi": hi,
                    "mae": float(np.mean(np.abs(yte - p))),
                    "r2": float(1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)),
                    "err_reduction_pct": float(100 * (1 - rmse(yte, p) / base)),
                    "circuit_evals_per_pred": circuits,
                    "n_params": int(npar.get(m, 0)), "fit_sec": npar.get(m + "_sec", 0.0),
                    "qk_bandwidth": npar.get("qk_ridge_quantum_bandwidth", np.nan)
                                    if m == "qk_ridge_quantum" else np.nan,
                    "n_test": len(yte), "extra": extra,
                    "kernel_scale": npar.get(m + "_scale", np.nan),
                    "kernel_alpha": npar.get(m + "_alpha", np.nan),
                    "kernel_cv_r2": npar.get(m + "_cv_r2", np.nan),
                    "kernel_candidates": npar.get(m + "_candidates", np.nan)})
    return out


def split_A(df, seed=SEED, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=df["circuit_id"]))


@dataclass
class SweepCase:
    df: pd.DataFrame
    train: np.ndarray
    test: np.ndarray
    experiment: str
    config: str
    split: str = "A_random_circuits"
    fast: bool = False
    feats: tuple = tuple(FEATS)
    arms: tuple = tuple(ALL_ARMS)
    extra: str = ""


def noise_strength_cases(directory=Path(".")):
    """Load the paired E1 datasets and construct one shared grouped split."""
    directory = Path(directory)
    alignment = ["circuit_id", "family", "depth", "observable",
                 "obs_weight", "n_1q", "n_2q", "ideal"]
    common_meta = ("wires", "per_family", "shots", "lambdas", "depths",
                   "circuit_seed", "shot_seed", "circuit_signature", "ideal_signature")
    datasets = []
    for mult in NOISE_LEVELS:
        path = directory / f"ds_noise{mult:.1f}x.csv"
        frame = pd.read_csv(path, float_precision="round_trip")
        meta = json.loads(path.with_name(f"{path.stem}_meta.json").read_text())
        required = {*common_meta, "generator", "noise_mult", "shot_spawn_key",
                    "n_rows", "n_circuits", "effective_noise"}
        if (not required.issubset(meta)
                or meta["generator"] != GENERATOR
                or meta["noise_mult"] != mult
                or meta["n_rows"] != len(frame)
                or meta["n_circuits"] != frame.circuit_id.nunique()
                or frame.duplicated(["circuit_id", "observable"]).any()):
            raise ValueError(f"Regenerate {path.name} with noise_sweep_data.py before running E1")
        datasets.append((mult, frame, meta))
    _, reference, reference_meta = datasets[0]
    if len({tuple(meta["shot_spawn_key"]) for _, _, meta in datasets}) != len(datasets):
        raise ValueError("E1 noise levels must have distinct shot substreams")
    for mult, frame, meta in datasets:
        if (any(meta[k] != reference_meta[k] for k in common_meta)
                or not frame[alignment].equals(reference[alignment])):
            raise ValueError(f"E1 circuit pool or ideal targets are misaligned at {mult:.1f}x")
    tr, te = split_A(reference)
    for mult, frame, meta in datasets:
        yield SweepCase(frame, tr, te, NOISE_EXPERIMENT, f"{mult:.1f}x", fast=True,
                        extra=f"p2_per_wire={meta['effective_noise']['p2']:.3f}")


def sweep_cases():
    """Yield all Chapter 5 sweep conditions."""
    yield from noise_strength_cases()
    main_df = pd.read_csv("benchmark_data.csv")

    rng = np.random.default_rng(SEED)
    for shots in [256, 1024, 4096, 0]:
        df = main_df.copy()
        for lam in ("1", "2", "3"):
            ex = df[f"exact_noisy_l{lam}"].to_numpy()
            if shots == 0:
                df[f"noisy_l{lam}"] = ex
            else:
                p = np.clip((1.0 + ex) / 2.0, 0.0, 1.0)
                df[f"noisy_l{lam}"] = 2.0 * rng.binomial(shots, p) / shots - 1.0
        tr, te = split_A(df)
        yield SweepCase(df, tr, te, "E2_shot_budget",
                        "analytic" if shots == 0 else str(shots), fast=True)

    for q, path in [(4, "ds_q4.csv"), (5, "benchmark_data.csv"), (6, "ds_q6.csv")]:
        df = main_df if q == 5 else pd.read_csv(path)
        tr, te = split_A(df)
        yield SweepCase(df, tr, te, "E3_qubit_count", f"{q}q", fast=q != 5)

    trB = main_df.index[main_df["depth"] <= 4].to_numpy()
    teB = main_df.index[main_df["depth"] >= 5].to_numpy()
    yield SweepCase(main_df, trB, teB, "E4_bounded_encoding", "train_depth<=4",
                    "B_depth_extrapolation")

    tr, te = split_A(main_df)
    tr_groups = np.unique(main_df.loc[tr, "circuit_id"])
    arms = ("no_mitigation", "zne_richardson2", "ridge", "hist_gbm", "mlp",
            *KERNEL_MODELS, "dr_vqc_quantum", "bounded_vqc_quantum")
    for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
        rs = np.random.default_rng(SEED)
        keep = rs.choice(tr_groups, size=max(8, int(len(tr_groups) * frac)), replace=False)
        tr_f = main_df.index[main_df["circuit_id"].isin(keep)].to_numpy()
        yield SweepCase(main_df, tr_f, te, "E5_learning_curve", str(frac), fast=True,
                        arms=arms, extra=f"n_train_circuits={len(keep)}")

    sets = {"lambda_only": tuple(LAM_FEATS),
            "single_noisy_plus_structure": ("noisy_l1", "depth", "n_2q", "obs_weight"),
            "full": tuple(FEATS)}
    for name, feats in sets.items():
        yield SweepCase(main_df, tr, te, "E6_feature_ablation", name, fast=True,
                        feats=feats, extra=f"n_features={len(feats)}",
                        arms=("no_mitigation", "ridge", "hist_gbm", "mlp",
                              *KERNEL_MODELS, "bounded_vqc_quantum"))

    for train_fam, test_fam in [("hea", "tfim"), ("tfim", "hea")]:
        trF = main_df.index[main_df["family"] == train_fam].to_numpy()
        teF = main_df.index[main_df["family"] == test_fam].to_numpy()
        yield SweepCase(main_df, trF, teF, "E7_cross_family", f"{train_fam}_to_{test_fam}",
                        "C_cross_family", fast=True)


def run_sweep_case(case):
    with threadpool_limits(limits=1):
        preds, npar, yte, gte = run_arms(case.df, case.train, case.test, feats=list(case.feats),
                                        arms=case.arms, fast=case.fast,
                                        label=f"{case.experiment} {case.config}")
    rows = score(preds, npar, yte, gte, case.experiment, case.config,
                 case.split, extra=case.extra)
    pair = {"experiment": case.experiment, "config": case.config, "split": case.split,
            **matched_kernel_difference(preds, yte, gte)}
    return rows, pair


def fit_sweep_cases(cases, workers):
    if workers == 1:
        yield from map(run_sweep_case, cases)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(run_sweep_case, cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-strength-only", action="store_true",
                        help="recompute E1 and update its rows in the existing sweep CSVs")
    parser.add_argument("--workers", type=int, default=1, help="parallel sweep fits")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    score_path = Path("sweep_results.csv")
    paired_path = Path("paired_kernel_sweeps.csv")
    previous = {}
    if args.noise_strength_only:
        for path in (score_path, paired_path):
            if not path.exists():
                parser.error(f"--noise-strength-only requires the existing {path.name}")
            before = pd.read_csv(path, float_precision="round_trip")
            previous[path] = before[before.experiment != NOISE_EXPERIMENT]

    cases = noise_strength_cases() if args.noise_strength_only else sweep_cases()
    rows, pairs = [], []
    for records, pair in fit_sweep_cases(cases, args.workers):
        rows.extend(records)
        pairs.append(pair)
        if not args.noise_strength_only:
            pd.DataFrame(rows).to_csv(score_path, index=False)
            pd.DataFrame(pairs).to_csv(paired_path, index=False)
    if args.noise_strength_only:
        for path, updated in ((score_path, rows), (paired_path, pairs)):
            combined = pd.concat([pd.DataFrame(updated), previous[path]], ignore_index=True)
            staging = path.with_suffix(".tmp")
            combined.to_csv(staging, index=False)
            staging.replace(path)
    print(pd.DataFrame(rows).groupby(["experiment", "config"]).size().to_string())


if __name__ == "__main__":
    main()
