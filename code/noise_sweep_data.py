"""Generate all E1 noise-strength datasets from one circuit pool."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import pandas as pd

from simulate import DEPTHS, GAMMA, LAMBDAS, P1, P2, build, make_qnode, sample


N_WIRES = 5
NOISE_LEVELS = (0.5, 1.0, 2.0, 4.0)
GENERATOR = "paired_noise_strength_v1"


def circuit_pool(per_family, seed):
    rng = np.random.default_rng(seed)
    return [
        (family, rng.uniform(-np.pi, np.pi, size=(DEPTHS[i % len(DEPTHS)], N_WIRES, 2)))
        for family in ("hea", "tfim")
        for i in range(per_family)
    ]


def circuit_signature(circuits):
    digest = sha256()
    for circuit_id, (family, params) in enumerate(circuits):
        digest.update(struct.pack("<QQQ", circuit_id, *params.shape[:2]))
        digest.update(family.encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(params, dtype="<f8").tobytes())
    return digest.hexdigest()


def expectation_values(rho, matrices):
    return [float(np.real(np.trace(matrix @ rho))) for matrix in matrices]


def _ideal_values(circuits):
    device, _, _, observables = build(N_WIRES)
    density, _ = make_qnode(device, N_WIRES, 1.0)
    matrices = tuple(observables.values())
    return np.asarray([
        expectation_values(density(params, family, 0.0), matrices)
        for family, params in circuits
    ])


def _simulate_noise_level(job):
    mult, circuits, ideal, shots, shot_stream = job
    device, observables, weights, matrices_by_name = build(N_WIRES)
    density, _ = make_qnode(device, N_WIRES, mult)
    matrices = tuple(matrices_by_name.values())
    rng = np.random.default_rng(shot_stream)
    rows = []
    for circuit_id, (family, params) in enumerate(circuits):
        depth = len(params)
        noisy = {
            lam: expectation_values(density(params, family, lam), matrices)
            for lam in LAMBDAS
        }
        for i, observable in enumerate(observables):
            row = {
                "circuit_id": circuit_id, "family": family, "depth": depth,
                "observable": observable, "obs_weight": weights[observable],
                "n_1q": depth * N_WIRES, "n_2q": depth * N_WIRES,
                "ideal": ideal[circuit_id, i],
            }
            for lam in LAMBDAS:
                row[f"exact_noisy_l{lam:g}"] = noisy[lam][i]
                row[f"noisy_l{lam:g}"] = sample(noisy[lam][i], rng, shots)
            rows.append(row)
        if (circuit_id + 1) % 100 == 0:
            print(f"E1 {mult:g}x: {circuit_id + 1}/{len(circuits)} circuits", flush=True)
    return pd.DataFrame(rows)


def generate_noise_sweep(outdir, *, per_family=250, shots=4096, circuit_seed=20260912,
                         shot_seed=20260913, workers=1):
    if per_family < 1 or shots < 1 or workers < 1:
        raise ValueError("per_family, shots and workers must be positive")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    circuits = circuit_pool(per_family, circuit_seed)
    ideal = _ideal_values(circuits)
    common = {
        "generator": GENERATOR,
        "wires": N_WIRES,
        "base_noise": {"p1": P1, "p2": P2, "gamma": GAMMA},
        "per_family": per_family,
        "shots": shots,
        "circuit_seed": circuit_seed,
        "shot_seed": shot_seed,
        "lambdas": list(LAMBDAS),
        "depths": list(DEPTHS),
        "n_circuits": len(circuits),
        "n_rows": ideal.size,
        "circuit_signature": circuit_signature(circuits),
        "ideal_signature": sha256(np.ascontiguousarray(ideal, dtype="<f8").tobytes()).hexdigest(),
    }
    # Assign children in fixed noise-level order, independently of simulation
    # scheduling and the binomial sampler's variable random-number consumption.
    streams = np.random.SeedSequence(shot_seed).spawn(len(NOISE_LEVELS))
    jobs = [(mult, circuits, ideal, shots, stream) for mult, stream in zip(NOISE_LEVELS, streams)]
    print(f"E1 shared pool: {len(circuits)} circuits, signature {common['circuit_signature']}", flush=True)
    if workers == 1:
        frames = list(map(_simulate_noise_level, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(NOISE_LEVELS))) as pool:
            frames = list(pool.map(_simulate_noise_level, jobs))

    for mult, stream, frame in zip(NOISE_LEVELS, streams, frames):
        base = outdir / f"ds_noise{mult:.1f}x"
        frame.to_csv(f"{base}.csv", index=False)
        metadata = {
            **common,
            "noise_mult": mult,
            "effective_noise": {name: value * mult for name, value in common["base_noise"].items()},
            "shot_spawn_key": list(stream.spawn_key),
        }
        Path(f"{base}_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    generate_noise_sweep(args.outdir, workers=args.workers)


if __name__ == "__main__":
    main()
