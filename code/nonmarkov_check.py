"""Verify that the production noise model is non-Markovian, and quantify it.

Uses the BLP measure (Breuer, Laine & Piilo, PRL 103, 210401 (2009)): a
dynamical map is non-Markovian if the trace distance between the reduced states
of two initial conditions is non-monotonic, i.e. information flows back from the
environment. We report

    N = sum_l max(0, D(l+1) - D(l))

over the layer index l, maximized over a set of initial state pairs, for the
production configuration and for the Markovian control (TLS switched off).
A strictly Markovian (CP-divisible) map gives N = 0 exactly.

Writes nonmarkovianity_check.json and, with --fig, the trace-distance figure.
"""
import argparse
import json

import numpy as np
import pennylane as qml

import simulate_realistic as sr

N_SYS = 5
N_LAYERS = 14
# Non-Markovianity is a property of the NOISE MAP, so it is characterised on an
# idling register (no gates), as in a device memory experiment. Running the
# probe through a scrambling circuit instead would measure unitary mixing: the
# Markovian control then also shows trace-distance revivals, which are not
# information back-flow.
PAIRS = {
    r"$|0\rangle$ vs $|1\rangle$ on $q_0$":
        (lambda: None, lambda: qml.PauliX(0)),
    r"$|+\rangle$ vs $|-\rangle$ on $q_0$":
        (lambda: qml.Hadamard(0), lambda: (qml.PauliX(0), qml.Hadamard(0))),
}


def trace_distance(a, b):
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(a - b))))


def idle_layer(n, c):
    """The device noise map during idling: no gates, only the incoherent
    channels, the always-on ZZ crosstalk, and the TLS collision."""
    for q in range(n):
        sr._d(c["p1"][q], q)
    for (a, b) in [(q, (q + 1) % n) for q in range(n)]:
        if c["J"][a] > 0:
            qml.IsingZZ(2 * float(c["J"][a]), wires=[a, b])
    for q in range(n):
        sr._ad(c["gamma"][q], q)
        sr._pd(c["gdeph"][q], q)
    for q in c["tls_wires"]:
        qml.IsingXX(2 * c["g_tls"], wires=[q, n])
        qml.IsingYY(2 * c["g_tls"], wires=[q, n])
    sr._ad(c["gamma_tls"], n)
    if c.get("tls_reset", False):             # memoryless control, as in simulate_realistic.layer
        qml.AmplitudeDamping(1.0, wires=n)


def trajectory(prep, cfg, n, n_layers, obs=(0,)):
    """Reduced state of the probe qubit after each idle layer."""
    dev = qml.device("default.mixed", wires=n + 1)
    out = []
    for L in range(1, n_layers + 1):
        @qml.qnode(dev)
        def circ():
            prep()           # constructing an op inside a QNode queues it
            for _ in range(L):
                idle_layer(n, cfg)
            return qml.density_matrix(wires=list(obs))
        out.append(np.asarray(circ()))
    return out


def measure(cfg, n=N_SYS, n_layers=N_LAYERS):
    res = {}
    for name, (p1, p2) in PAIRS.items():
        D = np.array([trace_distance(a, b) for a, b in
                      zip(trajectory(p1, cfg, n, n_layers),
                          trajectory(p2, cfg, n, n_layers))])
        res[name] = {"trace_distance": D.tolist(),
                     "blp": float(np.sum(np.clip(np.diff(D), 0, None))),
                     "n_revivals": int(np.sum(np.diff(D) > 1e-9)),
                     "d_min": float(D.min()), "d_max_after_min": float(D[D.argmin():].max())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()

    out = {"n_system_qubits": N_SYS, "n_layers": N_LAYERS,
           "definition": "N = sum_l max(0, D(l+1) - D(l)); 0 for a Markovian map",
           "reference": "Breuer, Laine & Piilo, PRL 103, 210401 (2009)"}

    for label, mk, rs in [("nonmarkovian_production", False, False), ("markovian_control", True, False),
                          ("reset_control", False, True)]:
        cfg = sr.make_cfg(N_SYS, seed=11, markovian=mk, tls_reset=rs)
        r = measure(cfg)
        out[label] = {"g_tls": float(cfg["g_tls"]), "pairs": r,
                      "blp_max": max(v["blp"] for v in r.values())}
        print(f"{label}: g_tls={cfg['g_tls']:.3f} BLP_max={out[label]['blp_max']:.4f}",
              flush=True)

    # coupling sweep: non-Markovianity should rise monotonically from zero
    sweep = []
    for g in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40):
        cfg = sr.make_cfg(N_SYS, seed=11, markovian=(g == 0.0))
        cfg["g_tls"] = g
        cfg["tls_wires"] = () if g == 0.0 else (0,)
        r = measure(cfg, n_layers=12)
        b = max(v["blp"] for v in r.values())
        sweep.append({"g_tls": g, "blp_max": b})
        print(f"  g_tls={g:.2f} -> BLP_max={b:.4f}", flush=True)
    out["coupling_sweep"] = sweep
    json.dump(out, open("nonmarkovianity_check.json", "w"), indent=2)

    if a.fig:
        # Same drawing as make_figures_realistic.py, which replots from the saved JSON.
        from make_figures_realistic import fig_nonmarkov
        fig_nonmarkov(out, a.fig)
        print("wrote", a.fig)


if __name__ == "__main__":
    main()
