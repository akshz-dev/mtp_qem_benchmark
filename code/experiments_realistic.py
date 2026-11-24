"""Experiments on the device-realistic, non-Markovian benchmark.

R1  main benchmark under BOTH noise-scaling methods (parameter scaling vs
    hardware-realizable layer-level unitary folding), on the random-circuit and
    depth-extrapolation splits
R2  memory: non-Markovian production model vs two paired controls, under BOTH
    scaling methods -- (i) the defect removed entirely ("markovian_control") and
    (ii) the defect kept but reset every layer ("reset_control"), which has the
    same per-layer damage but no memory (BLP measure exactly 0). Only (ii)
    isolates memory; (i) also removes the defect's loss.
R3  calibration drift: train at calibration A, deploy at calibration B
R4  shot-noise-limited quantum kernel: accuracy against total circuit executions,
    five independent shot-noise draws per budget
Every comparison between two conditions is also reported as a paired,
circuit-grouped bootstrap on the RMSE difference (paired_realistic.csv).

Reuses the model implementations in bench.py / experiments.py so the realistic
and idealized results come from identical code paths.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import bench
from bench import boot_rmse_ci, rmse
from experiments import qk_states_n, quantum_gram, run_arms, score
from stats_util import paired_diff

SEED = 7
PARAM_FEATS = ["noisy_l1", "noisy_l2", "noisy_l3", "depth", "n_2q", "obs_weight"]
FOLD_FEATS = ["fold_l1", "fold_l2", "fold_l3", "depth", "n_2q", "obs_weight"]
ARMS = ["no_mitigation", "zne_linear", "zne_richardson2", "zne_exponential",
        "ridge", "hist_gbm", "mlp", "qk_ridge_quantum", "dr_vqc_quantum",
        "bounded_vqc_quantum"]
QK_BW = (0.05, 0.15, 0.4, 1.0)


def split_random(df, seed=SEED):
    g = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    return next(g.split(df, groups=df["circuit_id"]))


def as_param(df, feats):
    """Present the chosen scaling columns under the names bench.py expects."""
    d = df.copy()
    for i, src in enumerate(feats[:3], start=1):
        d[f"noisy_l{i}"] = df[src]
    return d


def flush(rows, path="sweep_realistic.csv"):
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------- R4: shot-limited kernel
def shot_limited_kernel(df, tr, te, shots_list, rng, n_draws=5):
    """Estimate the fidelity Gram matrix from finite shots, as hardware must.

    Each entry k_ij = |<psi_i|psi_j>|^2 is measured by an inversion test with M
    shots, so the estimate is Binomial(M, k_ij)/M. Circuit executions consumed
    are M per distinct entry: M*(n_tr*(n_tr+1)/2 + n_tr*n_te).
    """
    sc = StandardScaler().fit(df.loc[tr, PARAM_FEATS].to_numpy())
    Xtr = sc.transform(df.loc[tr, PARAM_FEATS].to_numpy())
    Xte = sc.transform(df.loc[te, PARAM_FEATS].to_numpy())
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gtr, gte = df.loc[tr, "circuit_id"].to_numpy(), df.loc[te, "circuit_id"].to_numpy()
    base = rmse(yte, df.loc[te, "noisy_l1"].to_numpy())
    bw = 0.15
    Ptr, Pte = qk_states_n(Xtr, bw), qk_states_n(Xte, bw)
    K_exact = quantum_gram(Ptr)
    Kte_exact = quantum_gram(Pte, Ptr)
    n_tr, n_te = len(tr), len(te)
    out = []
    for M, draw in [(M, d) for M in shots_list for d in range(1 if M is None else n_draws)]:
        if M is None:
            Ktr, Kte, execs = K_exact, Kte_exact, 0
        else:
            iu = np.triu_indices(n_tr)
            vals = rng.binomial(M, np.clip(K_exact[iu], 0, 1)) / M
            Ktr = np.zeros_like(K_exact)
            Ktr[iu] = vals
            Ktr = Ktr + Ktr.T - np.diag(np.diag(Ktr))
            Kte = rng.binomial(M, np.clip(Kte_exact, 0, 1)) / M
            execs = int(M) * (n_tr * (n_tr + 1) // 2 + n_tr * n_te)
            # project to PSD: shot noise breaks positive semidefiniteness
            w, V = np.linalg.eigh(Ktr)
            Ktr = (V * np.clip(w, 0, None)) @ V.T
        g = GridSearchCV(KernelRidge(kernel="precomputed"),
                         {"alpha": np.logspace(-6, 1, 12)},
                         cv=GroupKFold(n_splits=4), n_jobs=1).fit(Ktr, ytr, groups=gtr)
        pred = np.clip(g.predict(Kte), -1.5, 1.5)
        lo, hi = boot_rmse_ci(yte, pred, gte, n=300)
        out.append({"experiment": "R4_shot_limited_kernel",
                    "config": "exact (simulator)" if M is None else f"{M} shots/entry",
                    "model": "qk_ridge_quantum", "kind": "hybrid quantum-classical",
                    "split": "A_random_circuits", "shots_per_entry": M or 0,
                    "draw": draw,
                    "circuit_executions": execs,
                    "kernel_circuits_per_prediction": n_tr,
                    "executions_per_prediction": int(M or 0) * n_tr,
                    "rmse": rmse(yte, pred),
                    "rmse_lo": lo, "rmse_hi": hi,
                    "err_reduction_pct": float(100 * (1 - rmse(yte, pred) / base)),
                    "unmitigated_rmse": base, "n_train": n_tr, "n_test": n_te})
        print(f"    M={M} draw={draw}: rmse={out[-1]['rmse']:.4f} execs={execs:.3g}", flush=True)
    return out


def _paired_rows(pa, ya, pb, yb, g, experiment, comparison, split, arms, seed):
    out = []
    for m in arms:
        if m in pa and m in pb:
            r = paired_diff(ya, np.clip(pa[m], -1.5, 1.5), yb, np.clip(pb[m], -1.5, 1.5), g, seed=seed)
            out.append({"experiment": experiment, "comparison": comparison, "split": split,
                        "model": m, **r})
    return out


def main():
    rng = np.random.default_rng(SEED)
    D = {k: pd.read_csv(f"ds_real_{k}.csv") for k in ("A", "B", "M", "R")}
    A, B = D["A"], D["B"]

    # The paired design requires that all datasets share circuits and ideal
    # values exactly; check it up front rather than half way through the run.
    meta = {k: json.load(open(f"ds_real_{k}_meta.json")) for k in D}
    sigs = {k: m["circuit_signature"] for k, m in meta.items()}
    isigs = {k: m["ideal_signature"] for k, m in meta.items()}
    assert len(set(sigs.values())) == 1, f"circuit sequences differ: {sigs}"
    assert len(set(isigs.values())) == 1, f"ideal values differ: {isigs}"
    print(f"paired design verified: circuit_signature={sigs['A']} "
          f"ideal_signature={isigs['A']}", flush=True)
    rows, pairs = [], []
    preds_store = {}

    def flush_all():
        flush(rows)
        pd.DataFrame(pairs).to_csv("paired_realistic.csv", index=False)

    # ---- R1: both scaling methods, both splits -----------------------------
    trA, teA = split_random(A)
    trD = A.index[A["depth"] <= 4].to_numpy()
    teD = A.index[A["depth"] >= 5].to_numpy()
    r1 = {}
    for label, feats in [("parameter_scaling", PARAM_FEATS),
                         ("unitary_folding", FOLD_FEATS)]:
        d = as_param(A, feats)
        for sp, (tr, te) in [("A_random_circuits", (trA, teA)),
                             ("B_depth_extrapolation", (trD, teD))]:
            pr, npr, yte, gte = run_arms(d, tr, te, arms=ARMS, fast=True,
                                         label=f"R1 {label} {sp}")
            rows += score(pr, npr, yte, gte, "R1_scaling_method", label, sp)
            r1[(label, sp)] = (pr, yte, gte)
            flush_all()
    for sp in ("A_random_circuits", "B_depth_extrapolation"):
        pf, yf, gf = r1[("unitary_folding", sp)]
        pp, yp, _ = r1[("parameter_scaling", sp)]
        pairs += _paired_rows(pf, yf, pp, yp, gf, "R1_scaling_method",
                              "folding_minus_parameter", sp, ARMS, seed=11)
        for lab in ("parameter_scaling", "unitary_folding"):
            p, y, g = r1[(lab, sp)]
            pairs += _paired_rows(p, y, {m: p["no_mitigation"] for m in p}, y, g, "R1_scaling_method",
                                  f"{lab}:arm_minus_unmitigated", sp, ARMS, seed=12)
            pairs += _paired_rows(p, y, {m: p["zne_linear"] for m in p}, y, g, "R1_scaling_method",
                                  f"{lab}:arm_minus_zne_linear", sp, ARMS, seed=13)
    flush_all()

    # ---- R2: memory, under both scaling methods ----------------------------
    # all four datasets share circuit_id, so one split serves them all
    for scal, feats in [("parameter_scaling", PARAM_FEATS), ("unitary_folding", FOLD_FEATS)]:
        res = {}
        for label, key in [("markovian_control", "M"), ("reset_control", "R"),
                           ("nonmarkovian_production", "A")]:
            d = as_param(D[key], feats)
            pr, npr, yte, gte = run_arms(d, trA, teA, arms=ARMS, fast=True,
                                         label=f"R2 {scal} {label}")
            rows += score(pr, npr, yte, gte, "R2_markovianity", f"{scal}:{label}",
                          "A_random_circuits")
            res[label] = (pr, yte, gte)
            flush_all()
        pA, yA, gA = res["nonmarkovian_production"]
        for ctrl in ("reset_control", "markovian_control"):
            pc, yc, _ = res[ctrl]
            pairs += _paired_rows(pA, yA, pc, yc, gA, "R2_markovianity",
                                  f"{scal}:production_minus_{ctrl}", "A_random_circuits", ARMS, seed=21)
        for label, (p, y, g) in res.items():
            pairs += _paired_rows(p, y, {m: p["no_mitigation"] for m in p}, y, g, "R2_markovianity",
                                  f"{scal}:{label}:arm_minus_unmitigated", "A_random_circuits", ARMS, seed=22)
        flush_all()

    # ---- R3: calibration drift ---------------------------------------------
    # A and B share circuit parameters (same seed), so rows are paired and the
    # only difference is the noise calibration.
    assert (A["circuit_id"].to_numpy() == B["circuit_id"].to_numpy()).all()
    assert np.allclose(A["ideal"], B["ideal"]), "ideal values must be identical"
    # one frame, A rows first then B rows, so train indices select calibration A
    # and test indices select calibration B on the very same circuits
    AB = pd.concat([A, B], ignore_index=True)
    off = len(A)
    r3 = {}
    for label, (tr_idx, te_idx) in {
            "no_drift_heldout": (trA, teA),
            "drift_heldout": (trA, teA + off),
            "drift_training_circuits": (trA, trA + off)}.items():
        pr, npr, yte, gte = run_arms(AB, tr_idx, te_idx, arms=ARMS, fast=True,
                                     label=f"R3 {label}")
        rows += score(pr, npr, yte, gte, "R3_calibration_drift", label,
                      "C_drift" if label.startswith("drift") else "A_random_circuits")
        r3[label] = (pr, yte, gte)
        flush_all()
    p1, y1, g1 = r3["drift_heldout"]
    p0, y0, _ = r3["no_drift_heldout"]
    pairs += _paired_rows(p1, y1, p0, y0, g1, "R3_calibration_drift",
                          "drift_minus_no_drift", "A_random_circuits", ARMS, seed=31)
    flush_all()

    # ---- R4: shot-limited kernel -------------------------------------------
    print("  R4 shot-limited quantum kernel", flush=True)
    rows += shot_limited_kernel(A, trA, teA, [100, 1000, 10000, 100000, None], rng)
    flush_all()

    out = pd.DataFrame(rows)
    out.to_csv("sweep_realistic.csv", index=False)
    print(out.groupby(["experiment", "config"]).size().to_string())


if __name__ == "__main__":
    main()
