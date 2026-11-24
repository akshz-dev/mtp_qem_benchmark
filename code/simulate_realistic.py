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

Emits the same flat schema as benchmark_data.csv plus fold_l* columns, so the
existing model code consumes it unchanged.

Usage:
  python simulate_realistic.py --out ds_real_A.csv --per-family 450
  python simulate_realistic.py --out ds_real_B.csv --per-family 450 --drift 0.30
  python simulate_realistic.py --out ds_real_M.csv --per-family 450 --markovian
"""
import argparse
import json

import numpy as np
import pennylane as qml

LAMBDAS = (1.0, 2.0, 3.0)
DEPTHS = (2, 3, 4, 5, 6)
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
        """fold: number of extra L L^dag pairs per layer (0 -> lam_fold=1,
        1 -> lam_fold=3 for that layer)."""
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
    """Layer-level unitary folding: lam=1 -> no folds; lam=3 -> every layer
    folded once; lam=2 -> half the layers folded (standard partial folding)."""
    if lam <= 1:
        return [0] * depth
    if lam >= 3:
        return [1] * depth
    k = int(round(depth * (lam - 1) / 2.0))
    return [1] * k + [0] * (depth - k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wires", type=int, default=5)
    ap.add_argument("--per-family", type=int, default=450)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--markovian", action="store_true")
    ap.add_argument("--tls-reset", action="store_true",
                    help="memoryless control: keep the defect coupling but reset the defect every layer")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n = a.wires
    nwires = n + 1                                  # +1 environment (TLS) wire
    obs = {"Z0": qml.PauliZ(0), "Z0Z1": qml.PauliZ(0) @ qml.PauliZ(1),
           "X0X1": qml.PauliX(0) @ qml.PauliX(1),
           "Z1Z2Z3": qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
           "Z_all": qml.prod(*[qml.PauliZ(i) for i in range(n)])}
    weight = {"Z0": 1, "Z0Z1": 2, "X0X1": 2, "Z1Z2Z3": 3, "Z_all": n}
    mats = {k: qml.matrix(v, wire_order=range(n)) for k, v in obs.items()}
    expvals = build(n, mats, nwires)

    cfg = make_cfg(n, seed=11, scale=a.scale, markovian=a.markovian, drift=a.drift,
                   tls_reset=a.tls_reset)
    ideal_cfg = dict(cfg)                           # noise-free reference
    for f in ("p1", "p2", "gamma", "gdeph", "eps", "J"):
        ideal_cfg[f] = np.zeros(n)
    ideal_cfg["g_tls"] = 0.0; ideal_cfg["gamma_tls"] = 0.0; ideal_cfg["tls_wires"] = ()

    # Two independent streams. This separation is load-bearing: Generator.binomial
    # uses rejection sampling, so it consumes a variable number of underlying draws
    # depending on its parameters. Sharing one generator with the circuit sampler
    # would make the circuit sequence depend on the noise configuration and silently
    # break the paired design across datasets.
    crng = np.random.default_rng(a.seed)          # circuit parameters
    srng = np.random.default_rng(a.seed + 1)      # shot noise
    rows, cid = [], 0
    for family in ("hea", "tfim"):
        for _ in range(a.per_family):
            depth = int(crng.choice(DEPTHS))
            th = crng.uniform(-np.pi, np.pi, (depth, n))
            ph = crng.uniform(-np.pi, np.pi, (depth, n))
            no_fold = [0] * depth

            ideal = expvals(th, ph, depth, ideal_cfg, family, no_fold)
            par = {lam: expvals(th, ph, depth, scaled(cfg, lam), family, no_fold)
                   for lam in LAMBDAS}
            fol = {lam: expvals(th, ph, depth, cfg, family, fold_pattern(depth, lam))
                   for lam in LAMBDAS}

            n_2q = depth * n
            for o in obs:
                row = {"circuit_id": cid, "family": family, "depth": depth,
                       "observable": o, "obs_weight": weight[o],
                       "n_1q": depth * n, "n_2q": n_2q, "ideal": ideal[o]}
                for lam in LAMBDAS:
                    row[f"exact_noisy_l{lam:g}"] = par[lam][o]
                    row[f"noisy_l{lam:g}"] = sample(par[lam][o], srng, a.shots)
                    row[f"exact_fold_l{lam:g}"] = fol[lam][o]
                    row[f"fold_l{lam:g}"] = sample(fol[lam][o], srng, a.shots)
                rows.append(row)
            cid += 1
            if cid % 50 == 0:
                print(f"  {a.out}: {cid} circuits", flush=True)

    import hashlib
    import pandas as pd
    df = pd.DataFrame(rows)
    sig = hashlib.sha1(
        df[["circuit_id", "family", "depth", "observable"]].to_csv(index=False).encode()
    ).hexdigest()[:16]
    ideal_sig = hashlib.sha1(np.round(df["ideal"].to_numpy(), 10).tobytes()).hexdigest()[:16]
    df.to_csv(a.out, index=False)
    meta = {"wires": n, "env_wires": 1, "markovian": bool(a.markovian), "tls_reset": bool(a.tls_reset),
            "drift": a.drift, "scale": a.scale, "shots": a.shots, "seed": a.seed,
            "per_family": a.per_family, "n_circuits": cid, "n_rows": len(df),
            "base_rates": {"p1": P1, "p2": P2, "gamma": GAMMA, "gdeph": GDEPH,
                           "eps_coherent": EPS_COH, "J_crosstalk": J_XTALK,
                           "g_tls": 0.0 if a.markovian else G_TLS,
                           "gamma_tls": 0.0 if a.markovian else GAMMA_TLS},
            "realised_rates": {k: np.asarray(cfg[k]).tolist()
                               for k in ("p1", "p2", "gamma", "gdeph", "eps", "J")},
            "circuit_signature": sig, "ideal_signature": ideal_sig,
            "scaling_methods": ["parameter scaling (noisy_l*)",
                                "layer-level unitary folding (fold_l*)"]}
    json.dump(meta, open(a.out.replace(".csv", "_meta.json"), "w"), indent=2)
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
