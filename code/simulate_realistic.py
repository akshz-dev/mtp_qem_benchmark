"""Device-realistic, non-Markovian benchmark generator.

Adds, relative to gen_data.py / simulate.py:
  * per-qubit inhomogeneous incoherent rates (device spread)
  * coherent over-rotation errors (systematic miscalibration)
  * always-on ZZ crosstalk between neighbours
  * non-Markovian dynamics via a coherently coupled two-level-system (TLS)
    defect: an XY-exchange collision model with an environment qubit that is
    never reset, so information flows back into the system
  * two noise-scaling methods for zero-noise extrapolation:
      - parameter scaling  (idealized; exact but not realizable on hardware)
      - layer-level unitary folding (hardware-realizable: L -> L L^dag L)
  * a drift option producing a second "calibration" of the same device

Emits the same flat schema as benchmark_data.csv plus fold_l* measurements and
fold_ratio_l* executed layer ratios. Both scaling methods use the native shot
estimate in noisy_l1.

Usage:
  python simulate_realistic.py --out ds_real_A.csv --per-family 450
  python simulate_realistic.py --out ds_real_B.csv --per-family 450 --drift 0.30
  python simulate_realistic.py --out ds_real_M.csv --per-family 450 --markovian
  python simulate_realistic.py --migrate-existing INPUT.csv --metadata INPUT_meta.json --out OUTPUT.csv
  python simulate_realistic.py --shots 0 --workers 4 --seed 2026112501 --out exact_pool.csv
  python simulate_realistic.py --resample-exact exact_pool.csv --shot-seed 2026112511 --out sampled_pool.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
import pennylane as qml

LAMBDAS = (1.0, 2.0, 3.0)
DEPTHS = (2, 3, 4, 5, 6)
FOLD_RATIO_COLUMNS = tuple(f"fold_ratio_l{lam:g}" for lam in LAMBDAS)
# base rates before per-qubit spread; two-qubit an order of magnitude above one-qubit
P1, P2, GAMMA, GDEPH = 0.0015, 0.012, 0.003, 0.004
EPS_COH, J_XTALK = 0.02, 0.015     # coherent over-rotation scale; crosstalk angle
G_TLS, GAMMA_TLS = 0.15, 0.01      # TLS exchange coupling; TLS relaxation


def make_cfg(n, seed=11, scale=1.0, markovian=False, drift=0.0, tls_reset=False):
    """Per-qubit noise configuration. drift>0 gives a later calibration."""
    rng = np.random.default_rng(seed)
    spread = lambda b: b * np.clip(1 + 0.35 * rng.standard_normal(n), 0.3, 2.0)
    c = dict(p1=scale * spread(P1), p2=scale * spread(P2),
             gamma=scale * spread(GAMMA), gdeph=scale * spread(GDEPH),
             eps=scale * EPS_COH * rng.standard_normal(n),
             J=scale * spread(J_XTALK),
             g_tls=0.0 if markovian else scale * G_TLS,
             gamma_tls=0.0 if markovian else scale * GAMMA_TLS,
             tls_wires=() if markovian else (0,),
             tls_reset=bool(tls_reset))
    if drift:
        d = np.random.default_rng(seed + 9999)
        for f in ("p1", "p2", "gamma", "gdeph", "J"):
            c[f] = c[f] * np.clip(1 + drift * d.standard_normal(n), 0.2, 3.0)
        c["eps"] = c["eps"] + drift * EPS_COH * d.standard_normal(n)
    return c


def scaled(c, lam):
    """Parameter noise scaling: every noise channel strength times lam."""
    out = dict(c)
    for f in ("p1", "p2", "gamma", "gdeph", "eps", "J"):
        out[f] = np.asarray(c[f]) * lam
    out["g_tls"] = c["g_tls"] * lam
    out["gamma_tls"] = c["gamma_tls"] * lam
    return out


def _d(p, w):
    p = float(np.clip(p, 0.0, 0.75))
    if p > 0:
        qml.DepolarizingChannel(p, wires=w)


def _ad(g, w):
    g = float(np.clip(g, 0.0, 1.0))
    if g > 0:
        qml.AmplitudeDamping(g, wires=w)


def _pd(g, w):
    g = float(np.clip(g, 0.0, 1.0))
    if g > 0:
        qml.PhaseDamping(g, wires=w)


def layer(th, ph, n, c, family, inverse=False):
    """One noisy layer. inverse=True applies the inverse unitary (still noisy),
    which is what unitary folding needs."""
    ring = [(q, (q + 1) % n) for q in range(n)]
    s = -1.0 if inverse else 1.0

    def rotations():
        for q in range(n):
            if family == "hea":
                # forward: RY then RZ; inverse: RZ^dag then RY^dag (reverse order)
                if inverse:
                    qml.RZ(s * ph[q] * (1 + c["eps"][q]), wires=q)
                    qml.RY(s * th[q] * (1 + c["eps"][q]), wires=q)
                else:
                    qml.RY(s * th[q] * (1 + c["eps"][q]), wires=q)
                    qml.RZ(s * ph[q] * (1 + c["eps"][q]), wires=q)
            else:
                qml.RX(s * th[q] * (1 + c["eps"][q]), wires=q)
            _d(c["p1"][q], q)

    def entangle():
        for (a, b) in (reversed(ring) if inverse else ring):
            if family == "hea":
                qml.CNOT(wires=[a, b])
            else:
                qml.IsingZZ(s * ph[a] * (1 + c["eps"][a]), wires=[a, b])
            _d(c["p2"][a], a)
            _d(c["p2"][b], b)

    if inverse:
        entangle(); rotations()
    else:
        rotations(); entangle()

    for (a, b) in ring:                       # always-on ZZ crosstalk (not inverted)
        if c["J"][a] > 0:
            qml.IsingZZ(2 * float(c["J"][a]), wires=[a, b])
    for q in range(n):                        # idle decoherence
        _ad(c["gamma"][q], q)
        _pd(c["gdeph"][q], q)
    for q in c["tls_wires"]:                  # non-Markovian TLS collision
        qml.IsingXX(2 * c["g_tls"], wires=[q, n])
        qml.IsingYY(2 * c["g_tls"], wires=[q, n])
    _ad(c["gamma_tls"], n)
    if c.get("tls_reset", False):             # memoryless control: fresh defect every layer
        qml.AmplitudeDamping(1.0, wires=n)    # full reset to |0>, discarding the stored excitation


def build(n, obs_mats, nwires):
    dev = qml.device("default.mixed", wires=nwires)

    @qml.qnode(dev)
    def state(th, ph, depth, c, family, fold):
        """fold gives the number of inverse-forward pairs per original layer."""
        for l in range(depth):
            layer(th[l], ph[l], n, c, family)
            for _ in range(fold[l]):
                layer(th[l], ph[l], n, c, family, inverse=True)
                layer(th[l], ph[l], n, c, family)
        return qml.density_matrix(wires=range(n))

    def expvals(th, ph, depth, c, family, fold):
        rho = np.asarray(state(th, ph, depth, c, family, fold))
        return {k: float(np.real(np.trace(m @ rho))) for k, m in obs_mats.items()}

    return expvals


def fold_pattern(depth, lam):
    """Fold the nearest integer number of layers needed for the requested scale."""
    if lam <= 1:
        return [0] * depth
    if lam >= 3:
        return [1] * depth
    k = int(round(depth * (lam - 1) / 2.0))
    return [1] * k + [0] * (depth - k)


def folded_layer_ratio(pattern):
    """Each fold adds one inverse layer and one forward layer."""
    depth = len(pattern)
    return (depth + 2 * sum(pattern)) / depth


def folding_metadata(source):
    return {
        "fold_ratio_columns": list(FOLD_RATIO_COLUMNS),
        "fold_ratio_definition": "executed layers / original layers",
        "native_shots_shared": True,
        "folding_data_provenance": source,
    }


def metadata_path(data_path):
    path = Path(data_path)
    return path.with_name(f"{path.stem}_meta.json")


def migrate_existing(data_path, meta_path, out_path):
    """Reuse stored measurements and preserve every other CSV value."""
    meta = json.loads(Path(meta_path).read_text())
    with open(data_path, newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = list(reader.fieldnames)

    ratios = {
        depth: [folded_layer_ratio(fold_pattern(depth, lam)) for lam in LAMBDAS]
        for depth in sorted({int(row["depth"]) for row in rows})
    }
    for row in rows:
        if float(row["exact_fold_l1"]) != float(row["exact_noisy_l1"]):
            raise ValueError(f"Native expectation mismatch for circuit {row['circuit_id']}, "
                             f"observable {row['observable']}")
        row["fold_l1"] = row["noisy_l1"]
        row.update(zip(FOLD_RATIO_COLUMNS, ratios[int(row["depth"])]))
    fields.extend(name for name in FOLD_RATIO_COLUMNS if name not in fields)
    with open(out_path, "w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    meta.update(folding_metadata("existing native parameter-scale shot estimate"))
    metadata_path(out_path).write_text(json.dumps(meta, indent=2) + "\n")


def resample_shots(df, shots, shot_seed):
    """Draw new Pauli means from the exact expectations using one native setting."""
    if shots < 1:
        raise ValueError("Shot resampling requires a positive shot count")
    columns = ["noisy_l1", "noisy_l2", "noisy_l3", "fold_l2", "fold_l3"]
    means = df[[f"exact_{column}" for column in columns]].to_numpy()
    if not np.array_equal(means[:, 0], df["exact_fold_l1"].to_numpy()):
        raise ValueError("Parameter scaling and folding have different native expectations")
    rng = np.random.default_rng(shot_seed)
    probabilities = np.clip((1.0 + means) / 2.0, 0.0, 1.0)
    sampled = df.copy()
    sampled[columns] = 2.0 * rng.binomial(shots, probabilities) / shots - 1.0
    sampled["fold_l1"] = sampled["noisy_l1"]
    return sampled


def circuit_inputs(n, per_family, circuit_seed):
    """Construct the original ordered circuit sequence independently of shots."""
    rng = np.random.default_rng(circuit_seed)
    circuits = []
    for family in ("hea", "tfim"):
        for _ in range(per_family):
            depth = int(rng.choice(DEPTHS))
            th = rng.uniform(-np.pi, np.pi, (depth, n))
            ph = rng.uniform(-np.pi, np.pi, (depth, n))
            circuits.append((len(circuits), family, depth, th, ph))
    return circuits


def simulate_rows(job):
    circuits, n, cfg, legacy_shots, legacy_shot_seed, output = job
    obs = {"Z0": qml.PauliZ(0), "Z0Z1": qml.PauliZ(0) @ qml.PauliZ(1),
           "X0X1": qml.PauliX(0) @ qml.PauliX(1),
           "Z1Z2Z3": qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
           "Z_all": qml.prod(*[qml.PauliZ(i) for i in range(n)])}
    weight = {"Z0": 1, "Z0Z1": 2, "X0X1": 2, "Z1Z2Z3": 3, "Z_all": n}
    mats = {k: qml.matrix(v, wire_order=range(n)) for k, v in obs.items()}
    expvals = build(n, mats, n + 1)

    ideal_cfg = dict(cfg)
    for field in ("p1", "p2", "gamma", "gdeph", "eps", "J"):
        ideal_cfg[field] = np.zeros(n)
    ideal_cfg["g_tls"] = 0.0
    ideal_cfg["gamma_tls"] = 0.0
    ideal_cfg["tls_wires"] = ()
    srng = np.random.default_rng(legacy_shot_seed)
    rows = []

    for cid, family, depth, th, ph in circuits:
        no_fold = [0] * depth
        ideal = expvals(th, ph, depth, ideal_cfg, family, no_fold)
        par = {lam: expvals(th, ph, depth, scaled(cfg, lam), family, no_fold)
               for lam in LAMBDAS}
        patterns = {lam: fold_pattern(depth, lam) for lam in LAMBDAS}
        fol = {lam: par[lam] if lam == 1.0 else expvals(th, ph, depth, cfg, family, patterns[lam])
               for lam in LAMBDAS}
        ratios = {name: folded_layer_ratio(patterns[lam])
                  for name, lam in zip(FOLD_RATIO_COLUMNS, LAMBDAS)}

        for o in obs:
            row = {"circuit_id": cid, "family": family, "depth": depth,
                   "observable": o, "obs_weight": weight[o],
                   "n_1q": depth * n, "n_2q": depth * n, "ideal": ideal[o]}
            for lam in LAMBDAS:
                row[f"exact_noisy_l{lam:g}"] = par[lam][o]
                row[f"noisy_l{lam:g}"] = sample(par[lam][o], srng, legacy_shots)
                row[f"exact_fold_l{lam:g}"] = fol[lam][o]
                if lam == 1.0:
                    row["fold_l1"] = row["noisy_l1"]
                    # Retain the legacy RNG slot for reproducible higher-scale draws.
                    sample(par[lam][o], srng, legacy_shots)
                else:
                    row[f"fold_l{lam:g}"] = sample(fol[lam][o], srng, legacy_shots)
            row.update(ratios)
            rows.append(row)
        if (cid + 1) % 50 == 0:
            print(f"  {output}: circuit {cid + 1}", flush=True)

    return rows


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--wires", type=int, default=5)
    ap.add_argument("--per-family", type=int, default=450)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel simulations when saving exact expectations")
    ap.add_argument("--shots", type=int, default=4096,
                    help="shots per measurement setting, or 0 to save exact expectations")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--markovian", action="store_true")
    ap.add_argument("--tls-reset", action="store_true",
                    help="memoryless control: keep the defect coupling but reset the defect every layer")
    ap.add_argument("--seed", "--circuit-seed", type=int, default=20260912)
    ap.add_argument("--shot-seed", type=int,
                    help="shot seed combined with the circuit seed for five measurement settings")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--migrate-existing", type=Path,
                        help="add executed ratios and share native shots in an archived CSV")
    source.add_argument("--resample-exact", type=Path,
                        help="draw new shots from stored exact expectation values")
    ap.add_argument("--metadata", type=Path, help="metadata for a supplied input CSV")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.migrate_existing:
        if a.shot_seed is not None:
            ap.error("--migrate-existing retains the archived shot estimates")
        migrate_existing(a.migrate_existing, a.metadata or metadata_path(a.migrate_existing), a.out)
        return
    if a.resample_exact:
        if a.shot_seed is None or a.shots < 1:
            ap.error("--resample-exact requires --shot-seed and a positive --shots value")
        meta = json.loads((a.metadata or metadata_path(a.resample_exact)).read_text())
        exact = pd.read_csv(a.resample_exact, float_precision="round_trip")
        stream = np.random.SeedSequence([a.shot_seed, meta["seed"]])
        resample_shots(exact, a.shots, stream).to_csv(a.out, index=False)
        meta.update(folding_metadata("independently resampled from exact expectations"))
        meta.update(shots=a.shots, shot_seed=a.shot_seed, shot_seed_components=stream.entropy)
        metadata_path(a.out).write_text(json.dumps(meta, indent=2) + "\n")
        return
    if a.metadata:
        ap.error("--metadata requires --migrate-existing or --resample-exact")
    if a.shot_seed is not None and a.shots < 1:
        ap.error("--shot-seed requires a positive --shots value")
    if a.workers < 1 or (a.workers > 1 and a.shots != 0):
        ap.error("--workers must be positive and requires --shots 0 when greater than one")

    n = a.wires
    cfg = make_cfg(n, seed=11, scale=a.scale, markovian=a.markovian, drift=a.drift,
                   tls_reset=a.tls_reset)
    circuits = circuit_inputs(n, a.per_family, a.seed)
    legacy_shots = a.shots if a.shot_seed is None else 0
    if a.workers == 1:
        rows = simulate_rows((circuits, n, cfg, legacy_shots, a.seed + 1, a.out))
    else:
        chunk_size = max(1, (len(circuits) + a.workers - 1) // a.workers)
        jobs = [(circuits[i:i + chunk_size], n, cfg, 0, a.seed + 1, a.out)
                for i in range(0, len(circuits), chunk_size)]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows = [row for chunk in pool.map(simulate_rows, jobs) for row in chunk]

    import hashlib
    df = pd.DataFrame(rows)
    if a.shot_seed is not None:
        stream = np.random.SeedSequence([a.shot_seed, a.seed])
        df = resample_shots(df, a.shots, stream)
    sig = hashlib.sha1(
        df[["circuit_id", "family", "depth", "observable"]].to_csv(index=False).encode()
    ).hexdigest()[:16]
    ideal_sig = hashlib.sha1(np.round(df["ideal"].to_numpy(), 10).tobytes()).hexdigest()[:16]
    df.to_csv(a.out, index=False)
    meta = {"wires": n, "env_wires": 1, "markovian": bool(a.markovian), "tls_reset": bool(a.tls_reset),
            "drift": a.drift, "scale": a.scale, "shots": a.shots, "seed": a.seed,
            "per_family": a.per_family, "n_circuits": len(circuits), "n_rows": len(df),
            "base_rates": {"p1": P1, "p2": P2, "gamma": GAMMA, "gdeph": GDEPH,
                           "eps_coherent": EPS_COH, "J_crosstalk": J_XTALK,
                           "g_tls": 0.0 if a.markovian else G_TLS,
                           "gamma_tls": 0.0 if a.markovian else GAMMA_TLS},
            "realised_rates": {k: np.asarray(cfg[k]).tolist()
                               for k in ("p1", "p2", "gamma", "gdeph", "eps", "J")},
            "circuit_signature": sig, "ideal_signature": ideal_sig,
            "scaling_methods": ["parameter scaling (noisy_l*)",
                                "layer-level unitary folding (fold_l*)"]}
    provenance = ("exact expectations (no finite-shot samples)" if a.shots == 0
                  else "generated with shared native shot estimate")
    meta.update(folding_metadata(provenance))
    if a.shot_seed is not None:
        meta.update(shot_seed=a.shot_seed, shot_seed_components=stream.entropy)
    metadata_path(a.out).write_text(json.dumps(meta, indent=2) + "\n")
    print("circuit_signature", sig, "ideal_signature", ideal_sig)
    print(a.out, df.shape,
          "mean|ideal-noisy_l1| =", round(float(np.mean(np.abs(df.ideal - df.noisy_l1))), 4),
          "mean|ideal-fold_l1| =", round(float(np.mean(np.abs(df.ideal - df.fold_l1))), 4))


def sample(mu, rng, shots):
    if shots <= 0:
        return float(mu)
    p = float(np.clip((1.0 + mu) / 2.0, 0.0, 1.0))
    return float(2.0 * rng.binomial(shots, p) / shots - 1.0)


if __name__ == "__main__":
    main()
