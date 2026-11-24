"""
qec_gen_data.py -- repetition-code memory-experiment data generation under the
PHENOMENOLOGICAL noise model, simulated directly in numpy.

SCOPE NOTE. The track uses the repetition code under phenomenological noise,
not the rotated surface code under circuit-level noise. Its space-time defect
graph is known analytically, so it can be simulated exactly in numpy with no
additional dependencies. See Chapter 6 of the thesis.

Model
  - Distance-d repetition code: d data qubits, d-1 ZZ stabilizers, T = d rounds.
  - Each data qubit independently bit-flips with probability p in each round
    (errors accumulate: the error state is the cumulative XOR).
  - Each stabilizer measurement independently reports the wrong value with
    probability p, independently in each round.
  - A final noiseless round of stabilizer values is appended (standard
    convention), obtained from a perfect readout of the data qubits.
  - Detectors are time differences of consecutive measured syndromes, so the
    detector array has shape (T+1, d-1).
  - The logical observable is the final error on data qubit 0: Z_0 is a valid
    logical-Z representative for the bit-flip repetition code, and an X-type
    error chain E flips it iff E[0] = 1.

Outputs (directory ./qec_data)
  cfg_d{d}_p{p}.npz   bit-packed detectors, observable flips, metadata
  manifest.json       full sweep description

Run:  python qec_gen_data.py
"""
import json
import os
import time

import numpy as np

DISTANCES = [3, 5, 7, 9]
ERROR_RATES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

N_SHOTS = 400_000        # per (d, p)
N_TRAIN = 200_000        # first block  -> training pool
N_TEST = 200_000         # second block -> held-out evaluation set
CHUNK = 50_000           # sampling chunk, bounds peak memory

MASTER_SEED = 20260912
OUTDIR = "qec_data"


def sample_config(d: int, p: float, n_shots: int, seed: int):
    """Return (detectors uint8 [n, (T+1)*(d-1)], observable flip uint8 [n])."""
    T = d
    n_stab = d - 1
    n_det = (T + 1) * n_stab
    det_out = np.empty((n_shots, n_det), dtype=np.uint8)
    obs_out = np.empty(n_shots, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    done = 0
    while done < n_shots:
        m = min(CHUNK, n_shots - done)
        # data-qubit flips per round, accumulated
        flips = (rng.random((m, T, d), dtype=np.float32) < p)
        E = np.cumsum(flips, axis=1, dtype=np.uint8) & 1        # (m, T, d)
        # true syndrome each round
        s_true = (E[:, :, :-1] ^ E[:, :, 1:])                    # (m, T, d-1)
        # noisy measurement
        readout = (rng.random((m, T, n_stab), dtype=np.float32) < p)
        meas = s_true ^ readout.astype(np.uint8)                 # (m, T, d-1)
        # appended noiseless final round from perfect data readout
        s_final = (E[:, -1, :-1] ^ E[:, -1, 1:])[:, None, :]     # (m, 1, d-1)
        m_all = np.concatenate([meas, s_final], axis=1)          # (m, T+1, d-1)
        # detectors = time differences, with an implicit all-zero round before t=0
        prev = np.concatenate(
            [np.zeros((m, 1, n_stab), dtype=np.uint8), m_all[:, :-1, :]], axis=1)
        det = m_all ^ prev
        det_out[done:done + m] = det.reshape(m, n_det)
        obs_out[done:done + m] = E[:, -1, 0]
        done += m
    return det_out, obs_out


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    assert N_TRAIN + N_TEST == N_SHOTS

    manifest = {
        "code": "repetition code (bit-flip), distance d, T = d rounds",
        "noise_model": "phenomenological: data-qubit flip probability p per "
                       "round; stabilizer measurement flip probability p per "
                       "round; final round noiseless",
        "scope_note": "Repetition code under phenomenological noise, not the "
                      "surface code under circuit-level noise (thesis Section 6.1).",
        "distances": DISTANCES,
        "error_rates": ERROR_RATES,
        "n_shots_per_config": N_SHOTS,
        "n_train_shots": N_TRAIN,
        "n_test_shots": N_TEST,
        "master_seed": MASTER_SEED,
        "observable": "final cumulative error on data qubit 0 (logical-Z "
                      "representative)",
        "configs": [],
    }

    for i, d in enumerate(DISTANCES):
        for j, p in enumerate(ERROR_RATES):
            t0 = time.time()
            seed = MASTER_SEED + 1000 * i + j
            det, obs = sample_config(d, p, N_SHOTS, seed)
            n_det = det.shape[1]
            np.savez_compressed(
                os.path.join(OUTDIR, f"cfg_d{d}_p{p}.npz"),
                det_packed=np.packbits(det, axis=1, bitorder="little"),
                obs=obs, n_detectors=n_det, d=d, p=p, rounds=d, seed=seed,
            )
            manifest["configs"].append({
                "d": d, "p": p, "rounds": d, "n_detectors": n_det,
                "n_stabilizers": d - 1, "detector_rounds": d + 1, "seed": seed,
                "mean_defects_per_shot": float(det.sum(1).mean()),
                "observable_flip_rate": float(obs.mean()),
                "gen_seconds": round(time.time() - t0, 2),
            })
            print(f"d={d} p={p}: {n_det} detectors, "
                  f"mean defects/shot={det.sum(1).mean():.2f}, "
                  f"raw obs flip rate={obs.mean():.4f}, "
                  f"{time.time()-t0:.1f}s", flush=True)

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {OUTDIR}/manifest.json ({len(manifest['configs'])} configs)")


if __name__ == "__main__":
    main()
