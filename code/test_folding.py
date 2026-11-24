"""Unit test: in the absence of noise, unitary folding must leave every
expectation value unchanged (L L^dag L = L). Run for both circuit families."""
import numpy as np, pennylane as qml
import simulate_realistic as sr
n = 5
obs = {"Z0": qml.PauliZ(0), "Z0Z1": qml.PauliZ(0) @ qml.PauliZ(1), "X0X1": qml.PauliX(0) @ qml.PauliX(1),
       "Z1Z2Z3": qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3), "Z_all": qml.prod(*[qml.PauliZ(i) for i in range(n)])}
mats = {k: qml.matrix(v, wire_order=range(n)) for k, v in obs.items()}
ev = sr.build(n, mats, n + 1)
zero = sr.make_cfg(n, scale=0.0, markovian=True)
rng = np.random.default_rng(0)
worst = {}
for fam in ("hea", "tfim"):
    w = 0.0
    for _ in range(6):
        depth = int(rng.integers(2, 7))
        th, ph = rng.uniform(-np.pi, np.pi, (depth, n)), rng.uniform(-np.pi, np.pi, (depth, n))
        a = ev(th, ph, depth, zero, fam, [0] * depth)
        b = ev(th, ph, depth, zero, fam, sr.fold_pattern(depth, 3.0))
        c = ev(th, ph, depth, zero, fam, sr.fold_pattern(depth, 2.0))
        w = max(w, max(abs(a[k] - b[k]) for k in a), max(abs(a[k] - c[k]) for k in a))
    worst[fam] = w
print("max |folded - unfolded| without noise:", {k: f"{v:.2e}" for k, v in worst.items()})
assert all(v < 1e-9 for v in worst.values()), worst
print("PASS")
