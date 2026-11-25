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
R1–R3 include the matched RBF and fidelity kernel-ridge fits. Paired,
circuit-grouped RMSE differences are reported in paired_realistic.csv.

Reuses the model implementations in bench.py / experiments.py so the realistic
and idealized results come from identical code paths.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from bench import boot_rmse_ci, rmse
from experiments import ZNE_SCALE_COLUMNS, run_arms, score
from shot_kernel import BANDWIDTH, SHOT_BUDGETS, SHOT_PROTOCOL, run_shot_study
from stats_util import paired_diff

SEED = 7
PARAM_FEATS = ["noisy_l1", "noisy_l2", "noisy_l3", "depth", "n_2q", "obs_weight"]
FOLD_FEATS = ["fold_l1", "fold_l2", "fold_l3", "depth", "n_2q", "obs_weight"]
FOLD_RATIO_COLUMNS = ["fold_ratio_l1", "fold_ratio_l2", "fold_ratio_l3"]
ARMS = ["no_mitigation", "zne_linear", "zne_richardson2", "zne_exponential",
        "ridge", "hist_gbm", "mlp", "rbf_krr", "qk_ridge_quantum",
        "dr_vqc_quantum", "bounded_vqc_quantum"]
RANDOM_SPLIT = "A_random_circuits"
DEPTH_SPLIT = "B_depth_extrapolation"
SCALING_FEATURES = {"parameter_scaling": PARAM_FEATS, "unitary_folding": FOLD_FEATS}


def split_random(df, seed=SEED):
    g = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    return next(g.split(df, groups=df["circuit_id"]))


def as_param(df, feats):
    """Expose the chosen measurements and the executed coordinates for folded ZNE."""
    if feats == FOLD_FEATS:
        if not set(FOLD_RATIO_COLUMNS).issubset(df.columns):
            raise ValueError("Folded ZNE requires the recorded fold_ratio_l1/l2/l3 columns")
        if not df["fold_l1"].equals(df["noisy_l1"]):
            raise ValueError("Folded and parameter-scaled arms must share the native shot estimate")
    d = df.copy()
    for i, src in enumerate(feats[:3], start=1):
        d[f"noisy_l{i}"] = df[src]
    if feats == FOLD_FEATS:
        d[ZNE_SCALE_COLUMNS] = df[FOLD_RATIO_COLUMNS].to_numpy()
    return d


# --------------------------------------------------------- R4: shot-limited kernel
def shot_limited_kernel(df, tr, te, *, workers=1):
    """Run the fidelity shot study on parameter-scaled Split A."""
    Xtr = df.loc[tr, PARAM_FEATS].to_numpy()
    Xte = df.loc[te, PARAM_FEATS].to_numpy()
    ytr, yte = df.loc[tr, "ideal"].to_numpy(), df.loc[te, "ideal"].to_numpy()
    gtr, gte = df.loc[tr, "circuit_id"].to_numpy(), df.loc[te, "circuit_id"].to_numpy()
    base = rmse(yte, df.loc[te, "noisy_l1"].to_numpy())
    n_tr, n_te = len(tr), len(te)
    out = []
    for row in run_shot_study(Xtr, ytr, gtr, Xte, SHOT_BUDGETS, seed=SEED, workers=workers):
        pred, fine_pred = np.clip(row.pop("predictions"), -1.5, 1.5).T
        shots = row["shots_per_entry"]
        error = rmse(yte, pred)
        lo, hi = boot_rmse_ci(yte, pred, gte, n=300)
        out.append({"experiment": "R4_shot_limited_kernel",
                    "config": f"{shots} shots/entry" if shots else "exact (simulator)",
                    "model": "qk_ridge_quantum", "kind": "hybrid quantum-classical",
                    "split": RANDOM_SPLIT, "kernel_scale": BANDWIDTH,
                    "circuit_executions": row["cv_circuit_executions"] + row["final_circuit_executions"],
                    "kernel_circuits_per_prediction": n_tr,
                    "executions_per_prediction": shots * n_tr,
                    "rmse": error,
                    "rmse_lo": lo, "rmse_hi": hi,
                    "fine_rmse": rmse(yte, fine_pred),
                    "err_reduction_pct": float(100 * (1 - error / base)),
                    "unmitigated_rmse": base, "n_train": n_tr, "n_test": n_te, **row})
        print(f"    shots={shots or 'exact'} draw={row['draw']}: RMSE={error:.4f}", flush=True)
    return out


def _paired_rows(pa, ya, pb, yb, g, experiment, comparison, split, arms, seed):
    out = []
    for m in arms:
        if m in pa and m in pb:
            r = paired_diff(ya, np.clip(pa[m], -1.5, 1.5), yb, np.clip(pb[m], -1.5, 1.5), g, seed=seed)
            out.append({"experiment": experiment, "comparison": comparison, "split": split,
                        "model": m, **r})
    return out


def assert_aligned(datasets):
    """Check that the noise controls can be compared row by row."""
    metadata = {key: json.loads(Path(f"ds_real_{key}_meta.json").read_text()) for key in datasets}
    for field in ("circuit_signature", "ideal_signature"):
        signatures = {key: value[field] for key, value in metadata.items()}
        if len(set(signatures.values())) != 1:
            raise ValueError(f"Paired data disagree on {field}: {signatures}")

    A = datasets["A"]
    keys = ["circuit_id", "family", "depth", "observable"]
    for name, other in datasets.items():
        if not A[keys].equals(other[keys]) or not np.allclose(A.ideal, other.ideal, rtol=0, atol=1e-12):
            raise ValueError(f"Calibration/condition {name} is misaligned with production A")
    print(f"paired design verified: circuit_signature={metadata['A']['circuit_signature']}", flush=True)


def build_exact_cases(datasets, trA, teA):
    """Identify distinct fits, sharing production fits across R1, R2 and R3."""
    A, B = datasets["A"], datasets["B"]
    trD = A.index[A.depth <= 4].to_numpy()
    teD = A.index[A.depth >= 5].to_numpy()
    AB = pd.concat([A, B], ignore_index=True)
    offset = len(A)
    # The first part is the R1/R2 production holdout; the other parts are R3.
    drift_test = np.concatenate([teA, teA + offset, trA + offset])
    cases = {}
    for scaling, features in SCALING_FEATURES.items():
        if scaling == "parameter_scaling":
            cases[(scaling, "A", RANDOM_SPLIT)] = AB, trA, drift_test
        else:
            cases[(scaling, "A", RANDOM_SPLIT)] = as_param(A, features), trA, teA
        cases[(scaling, "A", DEPTH_SPLIT)] = as_param(A, features), trD, teD
        for key in ("M", "R"):
            cases[(scaling, key, RANDOM_SPLIT)] = as_param(datasets[key], features), trA, teA
    return cases


def fit_exact_case(job):
    key, data, tr, te = job
    with threadpool_limits(limits=1):
        fit = run_arms(data, tr, te, arms=ARMS, fast=True, label=" / ".join(key))
    return key, fit


def summarize_exact_comparisons(fits, n_holdout):
    """Score all uses of the distinct fits and pair the held-out predictions."""
    rows, pairs = [], []

    def run_rows(result, experiment, config, split):
        preds, params, labels, groups = result
        rows.extend(score(preds, params, labels, groups, experiment, config, split))
        pairs.extend(_paired_rows(
            preds, labels, {"rbf_krr": preds["qk_ridge_quantum"]}, labels, groups,
            experiment, f"{config}:rbf_minus_fidelity", split, ("rbf_krr",), seed=14,
        ))

    def slice_test(result, selection):
        preds, params, labels, groups = result
        return ({m: p[selection] for m, p in preds.items()}, params, labels[selection], groups[selection])

    drift = fits[("parameter_scaling", "A", RANDOM_SPLIT)]
    production = fits | {("parameter_scaling", "A", RANDOM_SPLIT): slice_test(drift, slice(n_holdout))}

    # R1: noise scaling on the random-circuit and deeper-circuit holdouts.
    r1_name = "R1_scaling_method"
    for scaling in SCALING_FEATURES:
        for split in (RANDOM_SPLIT, DEPTH_SPLIT):
            run_rows(production[(scaling, "A", split)], r1_name, scaling, split)
    for split in (RANDOM_SPLIT, DEPTH_SPLIT):
        pf, _, yf, gf = production[("unitary_folding", "A", split)]
        pp, _, yp, _ = production[("parameter_scaling", "A", split)]
        pairs += _paired_rows(pf, yf, pp, yp, gf, r1_name,
                              "folding_minus_parameter", split, ARMS, seed=11)
        for scaling in SCALING_FEATURES:
            p, _, y, g = production[(scaling, "A", split)]
            for ref, tag, seed in (("no_mitigation", "unmitigated", 12), ("zne_linear", "zne_linear", 13)):
                pairs += _paired_rows(p, y, dict.fromkeys(p, p[ref]), y, g, r1_name,
                                      f"{scaling}:arm_minus_{tag}", split, ARMS, seed=seed)

    # R2: memory controls on the same production holdout for each scaling method.
    r2_name = "R2_markovianity"
    for scaling in SCALING_FEATURES:
        r2 = {name: production[(scaling, key, RANDOM_SPLIT)]
              for name, key in (("markovian_control", "M"), ("reset_control", "R"), ("nonmarkovian_production", "A"))}
        for label, result in r2.items():
            run_rows(result, r2_name, f"{scaling}:{label}", RANDOM_SPLIT)
        pA, _, yA, gA = r2["nonmarkovian_production"]
        for control in ("reset_control", "markovian_control"):
            pc, _, yc, _ = r2[control]
            pairs += _paired_rows(pA, yA, pc, yc, gA, r2_name,
                                  f"{scaling}:production_minus_{control}", RANDOM_SPLIT, ARMS, seed=21)
        for label, (p, _, y, g) in r2.items():
            pairs += _paired_rows(p, y, dict.fromkeys(p, p["no_mitigation"]), y, g, r2_name,
                                  f"{scaling}:{label}:arm_minus_unmitigated", RANDOM_SPLIT, ARMS, seed=22)

    # R3: one calibration-A fit, evaluated with and without drift.
    r3_name = "R3_calibration_drift"
    r3 = {
        "no_drift_heldout": slice_test(drift, slice(n_holdout)),
        "drift_heldout": slice_test(drift, slice(n_holdout, 2 * n_holdout)),
        "drift_training_circuits": slice_test(drift, slice(2 * n_holdout, None)),
    }
    for label, result in r3.items():
        run_rows(result, r3_name, label, "C_drift" if label.startswith("drift") else RANDOM_SPLIT)
    p1, _, y1, g1 = r3["drift_heldout"]
    p0, _, y0, _ = r3["no_drift_heldout"]
    pairs += _paired_rows(p1, y1, p0, y0, g1, r3_name,
                          "drift_minus_no_drift", RANDOM_SPLIT, ARMS, seed=31)
    return rows, pairs


def archived_shot_rows(path, A, trA, teA):
    """Reuse the separately run shot diagnostic when only R1–R3 are targeted."""
    archived = pd.read_csv(path, float_precision="round_trip")
    shots = archived.loc[archived.experiment == "R4_shot_limited_kernel"]
    required = {"shot_protocol", "model", "split", "shots_per_entry", "draw",
                "n_train", "n_test", "unmitigated_rmse", "kernel_alpha",
                "cv_circuit_executions", "final_circuit_executions", "circuit_executions",
                "executions_per_prediction", "fine_alpha", "fine_rmse"}
    if not required.issubset(shots.columns) or shots[list(required)].isna().any().any():
        raise ValueError("Saved R4 rows lack results from the fold-local PSD repair protocol")
    expected_budgets = {0: 1, 100: 5, 1000: 5, 10000: 5, 100000: 5}
    same_baseline = np.allclose(
        shots.unmitigated_rmse, rmse(A.loc[teA, "ideal"], A.loc[teA, "noisy_l1"]), rtol=0, atol=1e-12,
    )
    valid = (
        set(shots.shot_protocol) == {SHOT_PROTOCOL}
        and set(shots.model) == {"qk_ridge_quantum"}
        and set(shots.split) == {RANDOM_SPLIT}
        and shots.groupby("shots_per_entry").size().to_dict() == expected_budgets
        and not shots.duplicated(subset=["shots_per_entry", "draw"]).any()
        and shots.n_train.eq(len(trA)).all()
        and shots.n_test.eq(len(teA)).all()
        and shots.circuit_executions.eq(shots.cv_circuit_executions + shots.final_circuit_executions).all()
        and shots.executions_per_prediction.eq(shots.shots_per_entry * len(trA)).all()
        and same_baseline
    )
    if not valid:
        raise ValueError("Archived R4 rows do not match the expected Split A shot-budget diagnostic")
    print(f"retaining the {len(shots)} R4 shot-only diagnostic rows from {path}", flush=True)
    return shots.to_dict("records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="parallel fit workers")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reuse-shot-results", type=Path, help="retain compatible R4 rows from an existing CSV")
    mode.add_argument("--shot-only", action="store_true", help="replace only R4 in sweep_realistic.csv")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.shot_only:
        path = Path("sweep_realistic.csv")
        if not path.exists():
            parser.error("--shot-only requires the existing sweep_realistic.csv")
        scores = pd.read_csv(path, float_precision="round_trip")
        A = pd.read_csv("ds_real_A.csv")
        trA, teA = split_random(A)
        shot_rows = shot_limited_kernel(A, trA, teA, workers=args.workers)
        updated = pd.concat([scores[scores.experiment != "R4_shot_limited_kernel"],
                             pd.DataFrame(shot_rows)], ignore_index=True)
        staging = path.with_suffix(".tmp")
        updated.to_csv(staging, index=False)
        staging.replace(path)
        return

    datasets = {key: pd.read_csv(f"ds_real_{key}.csv") for key in ("A", "B", "M", "R")}
    assert_aligned(datasets)
    A = datasets["A"]
    trA, teA = split_random(A)
    shot_rows = archived_shot_rows(args.reuse_shot_results, A, trA, teA) if args.reuse_shot_results else None
    jobs = [(key, *case) for key, case in build_exact_cases(datasets, trA, teA).items()]
    if args.workers == 1:
        fits = dict(map(fit_exact_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fits = dict(pool.map(fit_exact_case, jobs))
    rows, pairs = summarize_exact_comparisons(fits, len(teA))
    if shot_rows is None:
        print("  R4 shot-limited quantum kernel", flush=True)
        shot_rows = shot_limited_kernel(A, trA, teA, workers=args.workers)
    out = pd.DataFrame(rows + shot_rows)
    out.to_csv("sweep_realistic.csv", index=False)
    pd.DataFrame(pairs).to_csv("paired_realistic.csv", index=False)
    print(out.groupby(["experiment", "config"]).size().to_string())


if __name__ == "__main__":
    main()
