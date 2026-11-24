"""Parametrized generator for the error-mitigation benchmark sweeps.

Generalizes gen_data.py over qubit count and noise strength so the thesis can
report noise-strength and system-size sweeps. Column schema is identical to
benchmark_data.csv, so the same model code consumes every dataset.

Usage: python simulate.py --wires 6 --noise-mult 1.0 --per-family 250 --out ds_q6.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
import pennylane as qml

LAMBDAS = (1.0, 2.0, 3.0)
P1, P2, GAMMA = 0.002, 0.02, 0.004   # base strengths at noise_mult = 1
DEPTHS = (2, 3, 4, 5, 6)


def build(n_wires):
    dev = qml.device("default.mixed", wires=n_wires)
    obs = {
        "Z0": qml.PauliZ(0),
        "Z0Z1": qml.PauliZ(0) @ qml.PauliZ(1),
        "X0X1": qml.PauliX(0) @ qml.PauliX(1),
        "Z1Z2Z3": qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
        "Z_all": qml.prod(*[qml.PauliZ(i) for i in range(n_wires)]),
    }
    weight = {"Z0": 1, "Z0Z1": 2, "X0X1": 2, "Z1Z2Z3": 3, "Z_all": n_wires}
    mats = {k: qml.matrix(v, wire_order=range(n_wires)) for k, v in obs.items()}
    return dev, obs, weight, mats


def make_qnode(dev, n_wires, mult):
    p1, p2, gam = P1 * mult, P2 * mult, GAMMA * mult

    def depol(p, wires):
        for w in wires:
            qml.DepolarizingChannel(min(p, 0.75), wires=w)

    def hea(theta, lam):
        for w in range(n_wires):
            qml.RY(theta[w, 0], wires=w)
            qml.RZ(theta[w, 1], wires=w)
            if lam > 0:
                depol(lam * p1, [w])
        for w in range(n_wires):
            pair = [w, (w + 1) % n_wires]
            qml.CNOT(wires=pair)
            if lam > 0:
                depol(lam * p2, pair)

    def tfim(theta, lam):
        for w in range(n_wires):
            qml.RX(theta[w, 0], wires=w)
            if lam > 0:
                depol(lam * p1, [w])
        for w in range(n_wires):
            pair = [w, (w + 1) % n_wires]
            qml.IsingZZ(theta[w, 1], wires=pair)
            if lam > 0:
                depol(lam * p2, pair)

    layers = {"hea": hea, "tfim": tfim}

    @qml.qnode(dev)
    def density(params, family, lam):
        layer = layers[family]
        for d in range(params.shape[0]):
            layer(params[d], lam)
            if lam > 0:
                for w in range(n_wires):
                    qml.AmplitudeDamping(min(lam * gam, 0.9), wires=w)
        return qml.density_matrix(wires=range(n_wires))

    return density, list(layers)


def sample(ev, rng, shots):
    p = np.clip((1.0 + ev) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(shots, p) / shots - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wires", type=int, default=5)
    ap.add_argument("--noise-mult", type=float, default=1.0)
    ap.add_argument("--per-family", type=int, default=250)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev, obs, weight, mats = build(a.wires)
    density, families = make_qnode(dev, a.wires, a.noise_mult)
    rng = np.random.default_rng(a.seed)

    def ev(rho):
        return {k: float(np.real(np.trace(M @ rho))) for k, M in mats.items()}

    rows, cid = [], 0
    for family in families:
        for i in range(a.per_family):
            depth = int(DEPTHS[i % len(DEPTHS)])
            params = rng.uniform(-np.pi, np.pi, size=(depth, a.wires, 2))
            ideal = ev(density(params, family, 0.0))
            noisy = {lam: ev(density(params, family, lam)) for lam in LAMBDAS}
            for o in obs:
                row = {"circuit_id": cid, "family": family, "depth": depth,
                       "observable": o, "obs_weight": weight[o],
                       "n_1q": depth * a.wires, "n_2q": depth * a.wires,
                       "ideal": ideal[o]}
                for lam in LAMBDAS:
                    row[f"exact_noisy_l{lam:g}"] = noisy[lam][o]
                    row[f"noisy_l{lam:g}"] = sample(noisy[lam][o], rng, a.shots)
                rows.append(row)
            cid += 1
            if cid % 200 == 0:
                print(f"  {a.out}: {cid} circuits", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    json.dump({"wires": a.wires, "noise_mult": a.noise_mult,
               "base_noise": {"p1": P1, "p2": P2, "gamma": GAMMA},
               "effective_noise": {"p1": P1 * a.noise_mult, "p2": P2 * a.noise_mult,
                                   "gamma": GAMMA * a.noise_mult},
               "per_family": a.per_family, "shots": a.shots, "seed": a.seed,
               "lambdas": list(LAMBDAS), "depths": list(DEPTHS),
               "n_circuits": cid, "n_rows": len(df)},
              open(a.out.replace(".csv", "_meta.json"), "w"), indent=2)
    print(a.out, df.shape, "mean|ideal-noisy_l1| =",
          round(float(np.mean(np.abs(df.ideal - df.noisy_l1))), 4))


if __name__ == "__main__":
    main()
