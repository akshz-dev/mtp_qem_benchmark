"""Generate an error-mitigation benchmark set: noisy -> ideal expectation values.

Row = (circuit, observable). Features = noise-scaled noisy expectation values
(the standard ZNE feature set) + circuit structure. Target = ideal expectation.

Noise model (base strengths, scaled by lambda):
  - single-qubit gate: DepolarizingChannel(p1)
  - two-qubit gate:    DepolarizingChannel(p2) on both wires
  - after every layer: AmplitudeDamping(gamma) on every wire (idle/T1 decay)
Noise scaling for ZNE is parameter scaling: (p1,p2,gamma) -> lambda*(p1,p2,gamma).
Finite sampling: each expectation is re-sampled from Binomial(shots, (1+<O>)/2).
"""
import json
import numpy as np
import pennylane as qml

N_WIRES = 5
LAMBDAS = (1.0, 2.0, 3.0)          # noise scale factors measured per circuit
P1, P2, GAMMA = 0.002, 0.02, 0.004  # base noise strengths
SHOTS = 4096
N_PER_FAMILY = 450
DEPTHS = (2, 3, 4, 5, 6)
SEED = 20260912

dev = qml.device("default.mixed", wires=N_WIRES)

OBSERVABLES = {
    "Z0": qml.PauliZ(0),
    "Z0Z1": qml.PauliZ(0) @ qml.PauliZ(1),
    "Z1Z2Z3": qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
    "Z_all": qml.prod(*[qml.PauliZ(i) for i in range(N_WIRES)]),
    "X0X1": qml.PauliX(0) @ qml.PauliX(1),
}
OBS_MAT = {k: qml.matrix(v, wire_order=range(N_WIRES)) for k, v in OBSERVABLES.items()}
OBS_WEIGHT = {"Z0": 1, "Z0Z1": 2, "Z1Z2Z3": 3, "Z_all": 5, "X0X1": 2}


def _noise(p, wires):
    for w in wires:
        qml.DepolarizingChannel(p, wires=w)


def hea_layer(theta, lam):
    """Hardware-efficient layer: RY,RZ on each wire + CNOT ring."""
    for w in range(N_WIRES):
        qml.RY(theta[w, 0], wires=w)
        qml.RZ(theta[w, 1], wires=w)
        _noise(min(lam * P1, 0.75), [w]) if lam > 0 else None
    for w in range(N_WIRES):
        pair = [w, (w + 1) % N_WIRES]
        qml.CNOT(wires=pair)
        if lam > 0:
            _noise(min(lam * P2, 0.75), pair)


def tfim_layer(theta, lam):
    """Trotter step of a transverse-field Ising evolution."""
    for w in range(N_WIRES):
        qml.RX(theta[w, 0], wires=w)
        if lam > 0:
            _noise(min(lam * P1, 0.75), [w])
    for w in range(N_WIRES):
        pair = [w, (w + 1) % N_WIRES]
        qml.IsingZZ(theta[w, 1], wires=pair)
        if lam > 0:
            _noise(min(lam * P2, 0.75), pair)


LAYERS = {"hea": hea_layer, "tfim": tfim_layer}


@qml.qnode(dev)
def density(params, family, lam):
    layer = LAYERS[family]
    for d in range(params.shape[0]):
        layer(params[d], lam)
        if lam > 0:
            for w in range(N_WIRES):
                qml.AmplitudeDamping(min(lam * GAMMA, 0.9), wires=w)
    return qml.density_matrix(wires=range(N_WIRES))


def expvals(rho):
    return {k: float(np.real(np.trace(M @ rho))) for k, M in OBS_MAT.items()}


def sample(ev, rng, shots=SHOTS):
    """Finite-shot estimate of a +-1-eigenvalue observable."""
    p = np.clip((1.0 + ev) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(shots, p) / shots - 1.0


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    cid = 0
    for family in LAYERS:
        for i in range(N_PER_FAMILY):
            depth = int(DEPTHS[i % len(DEPTHS)])
            params = rng.uniform(-np.pi, np.pi, size=(depth, N_WIRES, 2))
            ideal = expvals(density(params, family, 0.0))
            noisy = {lam: expvals(density(params, family, lam)) for lam in LAMBDAS}
            n_2q = depth * N_WIRES
            n_1q = depth * N_WIRES
            for obs in OBSERVABLES:
                row = {
                    "circuit_id": cid, "family": family, "depth": depth,
                    "observable": obs, "obs_weight": OBS_WEIGHT[obs],
                    "n_1q": n_1q, "n_2q": n_2q,
                    "ideal": ideal[obs],
                }
                for lam in LAMBDAS:
                    row[f"noisy_l{lam:g}"] = sample(noisy[lam][obs], rng)
                    row[f"exact_noisy_l{lam:g}"] = noisy[lam][obs]
                rows.append(row)
            cid += 1
            if cid % 100 == 0:
                print(f"{cid} circuits", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("benchmark_data.csv", index=False)
    meta = {
        "n_wires": N_WIRES, "lambdas": list(LAMBDAS),
        "base_noise": {"p1_1q_depol": P1, "p2_2q_depol": P2, "amp_damping_per_layer": GAMMA},
        "shots": SHOTS, "seed": SEED, "depths": list(DEPTHS),
        "families": list(LAYERS), "observables": list(OBSERVABLES),
        "noise_scaling": "noise-parameter scaling: (p1,p2,gamma) -> lambda*(p1,p2,gamma)",
        "n_circuits": int(cid), "n_rows": int(len(df)),
    }
    json.dump(meta, open("dataset_meta.json", "w"), indent=2)
    print(df.shape, df["ideal"].describe()[["mean", "std"]].to_dict())


if __name__ == "__main__":
    main()
