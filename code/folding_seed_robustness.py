"""Replicate the corrected R1 scaling comparison on new circuit pools and shot draws.

Each exact pool is evaluated on the reference random and depth holdouts, with
three matched inputs and structural features per method. Shot labels index
independent streams within each pool. Outputs summarize observed differences
by pool and shot draw, with paired circuit-bootstrap intervals within cells.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from experiments_realistic import (
    ARMS, DEPTH_SPLIT, RANDOM_SPLIT, SCALING_FEATURES, SEED, _paired_rows,
    as_param, fit_exact_case, score, split_random,
)
from simulate_realistic import metadata_path, resample_shots

EXPERIMENT = "R1_seed_replication"
KERNELS = ("rbf_krr", "qk_ridge_quantum")
KERNEL_CANDIDATES = 24
N_CV_FOLDS = 4
PROTOCOL = "executed-layer-ratios-shared-native-v1"


def _signature(*arrays):
    return sha256(np.concatenate(arrays).astype("<i8").tobytes()).hexdigest()


def _load_pool(path):
    meta = json.loads(metadata_path(path).read_text())
    pool = pd.read_csv(path, float_precision="round_trip")
    if (meta["shots"] != 0 or meta["scale"] != 1 or meta["drift"] != 0
            or meta["markovian"] or meta["tls_reset"] or meta["wires"] != 5):
        raise ValueError(f"Expected an exact Regime II production pool with five wires: {path}")
    if pool.circuit_id.nunique() != meta["n_circuits"] or len(pool) != meta["n_rows"]:
        raise ValueError(f"Pool size disagrees with metadata: {path}")
    if pool.duplicated(["circuit_id", "observable"]).any():
        raise ValueError(f"Repeated circuit-observable key: {path}")
    if pool.groupby("circuit_id").depth.nunique().ne(1).any():
        raise ValueError(f"A circuit has multiple depths: {path}")
    seed = int(meta["circuit_seed"] if "circuit_seed" in meta else meta["seed"])
    return seed, pool, {
        "circuit_seed": seed,
        "pool_file": path.name,
        "pool_sha256": sha256(path.read_bytes()).hexdigest(),
        "circuit_signature": meta["circuit_signature"],
        "ideal_signature": meta["ideal_signature"],
        "n_circuits": meta["n_circuits"],
        "n_rows": meta["n_rows"],
    }


def shared_splits(pool):
    """Choose group assignments once per pool, before using any shot data."""
    splits = {
        RANDOM_SPLIT: split_random(pool),
        DEPTH_SPLIT: (np.flatnonzero(pool.depth <= 4), np.flatnonzero(pool.depth >= 5)),
    }
    results = {}
    for split, (tr, te) in splits.items():
        train_groups = pool.circuit_id.iloc[tr].to_numpy()
        train_circuits = np.unique(train_groups)
        test_circuits = np.unique(pool.circuit_id.iloc[te])
        if len(train_circuits) < N_CV_FOLDS or not len(test_circuits):
            raise ValueError(f"Insufficient circuit groups for {split}")
        if np.intersect1d(train_circuits, test_circuits).size:
            raise ValueError(f"Overlapping train/test circuit groups for {split}")

        # The same GroupKFold constructor and training row order are used by both kernels.
        folds = np.empty(len(tr), dtype=int)
        fold_split = GroupKFold(n_splits=N_CV_FOLDS)
        for k, (_, validation) in enumerate(fold_split.split(tr, groups=train_groups)):
            folds[validation] = k
        tags = {
            "split": split,
            "n_train": len(tr),
            "n_test": len(te),
            "n_train_circuits": len(train_circuits),
            "n_test_circuits": len(test_circuits),
            "split_signature": _signature(train_circuits, [-1], test_circuits),
            "cv_signature": _signature(train_groups, [-1], folds),
        }
        results[split] = (tr, te, tags)
    return results


def _checkpoint(output, tags):
    return output / "checkpoints" / (
        f"{tags['circuit_seed']}-{tags['shot_seed']}-{tags['split']}-{tags['config']}.npz"
    )


def _inspect_checkpoint(path, tags):
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(saved["tags"].item()) != tags:
            raise ValueError(f"Checkpoint metadata disagrees with pool or run settings: {path}")


def _fit(job):
    path, tags, frame, tr, te = job
    key = (str(tags["circuit_seed"]), str(tags["shot_seed"]), tags["split"], tags["config"])
    _, (predictions, params, y, g) = fit_exact_case((key, frame, tr, te))
    if set(predictions) != set(ARMS):
        raise ValueError(f"Incomplete model fit for {key}")
    if any(params[f"{model}_candidates"] != KERNEL_CANDIDATES for model in KERNELS):
        raise ValueError(f"Unexpected kernel tuning budget for {key}")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, tags=json.dumps(tags), params=json.dumps(params, default=float),
        targets=y, groups=g, predictions=np.column_stack([predictions[arm] for arm in ARMS]),
    )
    temporary.replace(path)
    return path.name


def _load_fit(path, tags):
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(saved["tags"].item()) != tags:
            raise ValueError(f"Checkpoint metadata disagrees with pool or run settings: {path}")
        return (dict(zip(ARMS, saved["predictions"].T, strict=True)),
                json.loads(saved["params"].item()), saved["targets"], saved["groups"])


def _pool_cases(seed, pool_info, split_info, shot_seeds, shots):
    """Describe every fit and provide the shared outer train/test indices."""
    for shot_seed in shot_seeds:
        for split, (tr, te, tags) in split_info.items():
            for config in SCALING_FEATURES:
                yield {
                    **tags, "circuit_seed": seed, "shot_seed": shot_seed,
                    "shots": shots, "pool_sha256": pool_info["pool_sha256"],
                    "config": config, "protocol": PROTOCOL,
                }, tr, te


def _build_jobs(pools, shot_seeds, shots, output):
    for seed, pool, info, splits in pools:
        last_shot = None
        for tags, tr, te in _pool_cases(seed, info, splits, shot_seeds, shots):
            path = _checkpoint(output, tags)
            if path.exists():
                _inspect_checkpoint(path, tags)
                continue
            if tags["shot_seed"] != last_shot:
                last_shot = tags["shot_seed"]
                cell_rng = np.random.SeedSequence([last_shot, seed])
                measured = resample_shots(pool, shots, cell_rng)
                frames = {config: as_param(measured, features)
                          for config, features in SCALING_FEATURES.items()}
            yield path, tags, frames[tags["config"]], tr, te


def _score_cell(cell):
    """Score aligned predictions from the two methods and form paired contrasts."""
    row_tags, fits = cell
    p, _, y, groups = fits["parameter_scaling"]
    f, _, yf, gf = fits["unitary_folding"]
    if not (np.array_equal(y, yf) and np.array_equal(groups, gf)
            and np.array_equal(p["no_mitigation"], f["no_mitigation"])):
        raise ValueError(f"Unpaired test labels, circuit groups or native shots: {row_tags}")

    rows, comparisons = [], []
    split = row_tags["split"]
    for config, (pred, params, ideal, groups) in fits.items():
        rows.extend(score(pred, params, ideal, groups, EXPERIMENT, config, split))
        comparisons.extend(_paired_rows(
            pred, y, {"rbf_krr": pred["qk_ridge_quantum"]}, y, groups,
            EXPERIMENT, f"{config}:rbf_minus_fidelity", split, ("rbf_krr",), seed=14,
        ))
        for ref, tag, seed in (("no_mitigation", "unmitigated", 12),
                               ("zne_linear", "zne_linear", 13)):
            comparisons.extend(_paired_rows(
                pred, y, dict.fromkeys(pred, pred[ref]), y, groups,
                EXPERIMENT, f"{config}:arm_minus_{tag}", split, ARMS, seed=seed,
            ))
    comparisons.extend(_paired_rows(
        f, y, p, y, groups, EXPERIMENT, "folding_minus_parameter", split, ARMS, seed=11,
    ))
    return ([row_tags | row for row in rows], [row_tags | row for row in comparisons])


def summarize_pairs(pairs):
    """Report empirical spread and signs without treating cells as iid replicates."""
    keys = ["split", "comparison", "model"]
    classified = pairs.assign(
        negative=pairs["diff"] < 0, positive=pairs["diff"] > 0,
        resolved_negative=pairs["diff_hi"] < 0, resolved_positive=pairs["diff_lo"] > 0,
    )
    aggregations = dict(
        n_cells=("diff", "size"), diff_mean=("diff", "mean"), diff_min=("diff", "min"),
        diff_max=("diff", "max"), n_negative=("negative", "sum"), n_positive=("positive", "sum"),
        n_resolved_negative=("resolved_negative", "sum"),
        n_resolved_positive=("resolved_positive", "sum"),
    )
    by_pool = classified.groupby(keys + ["circuit_seed"], as_index=False).agg(
        n_shot_seeds=("shot_seed", "nunique"), **aggregations,
    )
    counts = classified.groupby(keys, as_index=False).agg(
        n_circuit_seeds=("circuit_seed", "nunique"), n_shot_seeds=("shot_seed", "nunique"),
        **{name: aggregation for name, aggregation in aggregations.items() if name != "diff_mean"},
    )
    pool_spread = by_pool.assign(
        mean_negative=by_pool["diff_mean"] < 0,
        mean_positive=by_pool["diff_mean"] > 0,
    ).groupby(keys, as_index=False).agg(
        diff_mean=("diff_mean", "mean"),
        pool_mean_min=("diff_mean", "min"), pool_mean_max=("diff_mean", "max"),
        n_pool_means_negative=("mean_negative", "sum"),
        n_pool_means_positive=("mean_positive", "sum"),
    )
    return counts.merge(pool_spread, on=keys, validate="one_to_one"), by_pool


def _load_scoring_cells(pools, shot_seeds, shots, output):
    staged = {}
    for seed, _, info, splits in pools:
        for tags, _, _ in _pool_cases(seed, info, splits, shot_seeds, shots):
            path = _checkpoint(output, tags)
            if not path.exists():
                raise ValueError(f"Missing fit checkpoint: {path}")
            key = (seed, tags["shot_seed"], tags["split"])
            if key not in staged:
                staged[key] = ({k: v for k, v in tags.items() if k != "config"}, {})
            staged[key][1][tags["config"]] = _load_fit(path, tags)
    return list(staged.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", action="append", type=Path, required=True,
                        help="exact production CSV with sibling metadata, repeat for each circuit seed")
    parser.add_argument("--shot-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fits-only", action="store_true", help="write reusable prediction checkpoints")
    mode.add_argument("--summarize-only", action="store_true", help="assemble completed checkpoints")
    args = parser.parse_args()
    if args.workers < 1 or args.shots < 1 or min(args.shot_seeds) < 0:
        parser.error("workers and shots must be positive and shot seeds nonnegative")
    if len(set(args.shot_seeds)) != len(args.shot_seeds):
        parser.error("shot seeds must be distinct")
    pools = [_load_pool(path) for path in args.pool]
    pools.sort(key=lambda pool: pool[0])
    if len({seed for seed, _, _ in pools}) != len(pools):
        parser.error("circuit seeds must be distinct")
    if len({info["ideal_signature"] for _, _, info in pools}) != len(pools):
        parser.error("the circuit pools must have distinct ideal signatures")
    if set(args.shot_seeds).intersection(seed for seed, _, _ in pools):
        parser.error("circuit seeds and shot seed labels must be disjoint")
    pools = [(seed, pool, info, shared_splits(pool)) for seed, pool, info in pools]

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        (args.output / "checkpoints").mkdir(exist_ok=True)
        jobs = _build_jobs(pools, args.shot_seeds, args.shots, args.output)
        if args.workers == 1:
            for name in map(_fit, jobs):
                print(f"saved {name}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for name in executor.map(_fit, jobs):
                    print(f"saved {name}", flush=True)
    if args.fits_only:
        return

    cells = _load_scoring_cells(pools, args.shot_seeds, args.shots, args.output)
    if args.workers == 1:
        results = list(map(_score_cell, cells))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_score_cell, cells))
    scores = pd.DataFrame(row for score_rows, _ in results for row in score_rows)
    pairs = pd.DataFrame(row for _, paired_rows in results for row in paired_rows)
    summary, by_pool = summarize_pairs(pairs)
    for suffix, frame in (("scores", scores), ("pairs", pairs), ("summary", summary),
                          ("pool_summary", by_pool)):
        frame.to_csv(args.output / f"folding_seed_{suffix}.csv", index=False)

    manifest = {
        "experiment": EXPERIMENT, "protocol": PROTOCOL,
        "circuit_pools": [
            {**info, "splits": [tags for _, _, tags in splits.values()]}
            for _, _, info, splits in pools
        ],
        "shot_seeds": args.shot_seeds, "shots_per_observable_setting": args.shots,
        "shot_stream": "numpy SeedSequence([shot_seed, circuit_seed]) with PCG64",
        "split_seed": SEED, "outer_random_test_fraction": 0.30,
        "depth_training_max": 4, "depth_test_min": 5,
        "inner_group_folds": N_CV_FOLDS, "kernel_candidates": KERNEL_CANDIDATES,
        "scaling_methods": list(SCALING_FEATURES), "models": ARMS,
        "paired_bootstrap_replicates": 1000,
        "aggregate_uncertainty": "empirical spread across cells and across circuit pool means",
        "n_pool_shot_cells": len(pools) * len(args.shot_seeds),
        "n_scored_holdouts": len(cells),
        "n_fit_checkpoints": len(cells) * len(SCALING_FEATURES),
    }
    (args.output / "folding_seed_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(scores)} scores and {len(pairs)} contrasts across {len(cells)} holdouts", flush=True)


if __name__ == "__main__":
    main()
