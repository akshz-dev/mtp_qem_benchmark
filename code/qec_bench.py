"""
qec_bench.py -- benchmark classical matching decoders against ML syndrome
decoders on the repetition-code / phenomenological-noise data produced by
qec_gen_data.py.

SCOPE NOTE. The benchmark uses the repetition code under phenomenological noise,
not the surface code under circuit-level noise (thesis Section 6.1). Every number
produced here comes from the numpy simulation in qec_gen_data.py.

Decoders
  MWPM        exact minimum-weight perfect matching with boundary, on the 1+1D
              space-time defect graph. Blossom solver: networkx.max_weight_matching
              (Galil's exact algorithm) on negated weights, validated against
              brute-force enumeration.
  Greedy      greedy nearest-available matching (weaker classical baseline).
  LogReg      logistic regression on the flattened detector syndrome.
  MLP         feed-forward network on the flattened detector syndrome (sklearn).
  VQC-PCA     data-reuploading variational quantum circuit on PCA-compressed
              syndromes with a logistic head (PennyLane).

Metric: logical error rate per shot on a disjoint held-out shot set, Wilson 95% CI.

Run:  python qec_bench.py
"""
import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "qec_data"
RESULTS_CSV = "qec_decoder_results.csv"
SEED = 7

# ---- compute budgets; every reduction is recorded in the results table ------
MWPM_TIME_BUDGET = 150.0     # s per config for the exact matching decoder
GREEDY_TIME_BUDGET = 60.0
MWPM_MIN_SHOTS = 20_000
MLP_MAX_TRAIN = 200_000
VQC_MAX_TRAIN = 15_000
VQC_MAX_TEST = 50_000
N_PCA = 6
MLP_HIDDEN = (128, 64)
MLP_MAX_ITER = 150
MLP_NO_CHANGE = 12
VQC_LAYERS = 3
VQC_EPOCHS = 20
VQC_BATCH = 256


# ============================================================================
# statistics
# ============================================================================
def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


# ============================================================================
# matching decoders on the 1+1D defect graph
# ============================================================================
# A defect at flat detector index k sits at (t, j) = (k // n_stab, k % n_stab),
# where j indexes the stabilizer between data qubits j and j+1.
#   defect-defect weight  : |t1-t2| + |j1-j2|      (Manhattan / graph distance)
#   defect-left-boundary  : j + 1                  (flip qubits 0..j)
#   defect-right-boundary : d - 1 - j              (flip qubits j+1..d-1)
# Only a left-boundary connection flips data qubit 0, so the decoder's predicted
# observable is  (# defects matched to the LEFT boundary) mod 2.

def _coords(idx, n_stab):
    return idx // n_stab, idx % n_stab


def mwpm_predict(idx, n_stab, d):
    """Exact MWPM with boundary. Returns (predicted observable bit, cost)."""
    import networkx as nx

    n = len(idx)
    if n == 0:
        return 0, 0
    t, j = _coords(idx, n_stab)
    left = j + 1
    right = d - 1 - j
    bcost = np.minimum(left, right)
    bleft = left < right              # cheaper boundary is the left one

    if n == 1:
        return int(bleft[0]), int(bcost[0])
    if n == 2:
        pair = abs(int(t[0]) - int(t[1])) + abs(int(j[0]) - int(j[1]))
        both = int(bcost[0] + bcost[1])
        if pair <= both:
            return 0, pair
        return int(bleft[0]) ^ int(bleft[1]), both

    # general case: 2n-node graph (n defects + n private boundary nodes)
    G = nx.Graph()
    for a in range(n):
        for b in range(a + 1, n):
            w = abs(int(t[a]) - int(t[b])) + abs(int(j[a]) - int(j[b]))
            G.add_edge(a, b, weight=-float(w))
        G.add_edge(a, n + a, weight=-float(bcost[a]))
    for a in range(n):
        for b in range(a + 1, n):
            G.add_edge(n + a, n + b, weight=0.0)

    mate = nx.max_weight_matching(G, maxcardinality=True)
    parity, cost = 0, 0
    for u, v in mate:
        if u > v:
            u, v = v, u
        if v >= n and u < n:          # defect u sent to a boundary
            cost += int(bcost[u])
            parity ^= int(bleft[u])
        elif v < n:                   # defect-defect pair
            cost += abs(int(t[u]) - int(t[v])) + abs(int(j[u]) - int(j[v]))
    return parity, cost


def greedy_predict(idx, n_stab, d):
    """Greedy nearest-available matching. Returns (observable bit, cost)."""
    n = len(idx)
    if n == 0:
        return 0, 0
    t, j = _coords(idx, n_stab)
    left = (j + 1).astype(int)
    right = (d - 1 - j).astype(int)
    bcost = np.minimum(left, right)
    bleft = left < right
    alive = np.ones(n, dtype=bool)
    parity, cost = 0, 0
    while alive.any():
        live = np.flatnonzero(alive)
        best, act = None, None
        for ai in range(len(live)):
            a = live[ai]
            if best is None or bcost[a] < best:
                best, act = int(bcost[a]), ("B", a, -1)
            for b in live[ai + 1:]:
                w = abs(int(t[a]) - int(t[b])) + abs(int(j[a]) - int(j[b]))
                if w < best:
                    best, act = w, ("P", a, b)
        kind, a, b = act
        cost += best
        if kind == "B":
            parity ^= int(bleft[a])
            alive[a] = False
        else:
            alive[a] = False
            alive[b] = False
    return parity, cost


def brute_force_cost(idx, n_stab, d):
    """Exhaustive minimum over all partial matchings (unmatched -> boundary)."""
    n = len(idx)
    t, j = _coords(idx, n_stab)
    left = (j + 1).astype(int)
    right = (d - 1 - j).astype(int)
    bcost = np.minimum(left, right)

    def rec(remaining):
        if not remaining:
            return 0
        a = remaining[0]
        rest = remaining[1:]
        best = int(bcost[a]) + rec(rest)              # send a to a boundary
        for k in range(len(rest)):
            b = rest[k]
            w = abs(int(t[a]) - int(t[b])) + abs(int(j[a]) - int(j[b]))
            best = min(best, w + rec(rest[:k] + rest[k + 1:]))
        return best

    return rec(tuple(range(n)))


def validate_matching(n_trials=400, seed=11):
    """Check the blossom MWPM cost against brute force on small instances."""
    rng = np.random.default_rng(seed)
    n_ok = n_bad = 0
    greedy_opt = 0
    worst = []
    for _ in range(n_trials):
        d = int(rng.choice([3, 5, 7, 9]))
        n_stab = d - 1
        T1 = d + 1
        n_det = T1 * n_stab
        n_def = int(rng.integers(1, min(8, n_det) + 1))
        idx = rng.choice(n_det, size=n_def, replace=False)
        c_mwpm = mwpm_predict(idx, n_stab, d)[1]
        c_bf = brute_force_cost(idx, n_stab, d)
        c_gr = greedy_predict(idx, n_stab, d)[1]
        if c_mwpm == c_bf:
            n_ok += 1
        else:
            n_bad += 1
            worst.append((d, sorted(idx.tolist()), c_mwpm, c_bf))
        if c_gr == c_bf:
            greedy_opt += 1
    return {"n_trials": n_trials, "mwpm_matches_bruteforce": n_ok,
            "mwpm_mismatches": n_bad, "mismatch_examples": worst[:5],
            "greedy_optimal_fraction": greedy_opt / n_trials}


def run_matching(det_te, y_te, n_stab, d, fn, time_budget):
    """Apply a matching decoder shot-by-shot with a wall-clock budget."""
    cache = {}
    n_total = len(det_te)
    errs = 0
    t0 = time.time()
    used = 0
    for i in range(n_total):
        row = det_te[i]
        key = row.tobytes()
        pred = cache.get(key)
        if pred is None:
            pred = fn(np.flatnonzero(row), n_stab, d)[0]
            cache[key] = pred
        errs += int(pred != y_te[i])
        used = i + 1
        if (i & 1023) == 1023 and (time.time() - t0) > time_budget \
                and used >= MWPM_MIN_SHOTS:
            break
    dt = time.time() - t0
    return errs, used, dt, 0, len(cache)


# ============================================================================
# ML decoders
# ============================================================================
def run_logreg(det_tr, y_tr, det_te, y_te):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, n_jobs=1, C=1.0)
    clf.fit(det_tr, y_tr)
    t0 = time.time()
    pred = clf.predict(det_te)
    dt = time.time() - t0
    return (int((pred != y_te).sum()), len(det_te), dt,
            int(clf.coef_.size + clf.intercept_.size), None)


def run_logreg_pca(det_tr, y_tr, det_te, y_te, pca_state):
    """Classical control on exactly the features the VQC receives. Isolates the
    cost of PCA compression from the behaviour of the quantum model."""
    from sklearn.linear_model import LogisticRegression

    pca = pca_state[0]
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(pca.transform(det_tr), y_tr)
    Zte = pca.transform(det_te)
    t0 = time.time()
    pred = clf.predict(Zte)
    dt = time.time() - t0
    return (int((pred != y_te).sum()), len(det_te), dt,
            int(clf.coef_.size + clf.intercept_.size), None)


def run_mlp(det_tr, y_tr, det_te, y_te):
    from sklearn.neural_network import MLPClassifier

    n = min(MLP_MAX_TRAIN, len(det_tr))
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation="relu",
                        alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                        max_iter=MLP_MAX_ITER, early_stopping=True,
                        n_iter_no_change=MLP_NO_CHANGE,
                        validation_fraction=0.1, random_state=SEED)
    clf.fit(det_tr[:n], y_tr[:n])
    t0 = time.time()
    pred = clf.predict(det_te)
    dt = time.time() - t0
    npar = sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    return int((pred != y_te).sum()), len(det_te), dt, int(npar), clf.n_iter_


def run_vqc(det_tr, y_tr, det_te, y_te, pca_state):
    """Data-reuploading VQC on N_PCA PCA features with a logistic head."""
    import pennylane as qml
    from pennylane import numpy as pnp

    pca, mu, sd = pca_state
    ntr = min(VQC_MAX_TRAIN, len(det_tr))
    nte = min(VQC_MAX_TEST, len(det_te))
    Ztr = np.clip((pca.transform(det_tr[:ntr]) - mu) / sd, -3, 3) * (np.pi / 3)
    Zte = np.clip((pca.transform(det_te[:nte]) - mu) / sd, -3, 3) * (np.pi / 3)
    ytr = y_tr[:ntr].astype(np.float64)
    yte = y_te[:nte]

    nq = N_PCA
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(x, w):
        for L in range(VQC_LAYERS):
            for q in range(nq):
                qml.RY(x[..., q], wires=q)
                qml.RZ(x[..., q], wires=q)
            for q in range(nq):
                qml.RY(w[L, q, 0], wires=q)
                qml.RZ(w[L, q, 1], wires=q)
            for q in range(nq):
                qml.CNOT(wires=[q, (q + 1) % nq])
        return qml.expval(qml.PauliZ(0))

    rng = np.random.default_rng(SEED)
    w0 = pnp.array(0.1 * rng.standard_normal((VQC_LAYERS, nq, 2)), requires_grad=True)
    h0 = pnp.array([4.0, 0.0], requires_grad=True)

    def logits(x, w, h):
        return h[0] * circuit(x, w) + h[1]

    def cost(w, h, xb, yb):
        z = logits(xb, w, h)
        return pnp.mean(pnp.logaddexp(0.0, z) - yb * z)   # BCE with logits

    ns = int(0.9 * ntr)
    Xtr, Ytr = Ztr[:ns], ytr[:ns]
    Xva, Yva = Ztr[ns:], ytr[ns:]
    opt = qml.AdamOptimizer(0.05)
    w, h = w0, h0
    best, best_wh, bad = np.inf, (w0, h0), 0
    ep = 0
    for ep in range(VQC_EPOCHS):
        perm = rng.permutation(ns)
        for i in range(0, ns, VQC_BATCH):
            b = perm[i:i + VQC_BATCH]
            xb = pnp.array(Xtr[b], requires_grad=False)
            yb = pnp.array(Ytr[b], requires_grad=False)
            w, h = opt.step(lambda ww, hh: cost(ww, hh, xb, yb), w, h)
        vz = logits(pnp.array(Xva, requires_grad=False), w, h)
        vler = float(np.mean((np.asarray(vz) > 0).astype(np.float64) != Yva))
        if vler < best - 1e-6:
            best, best_wh, bad = vler, (w.copy(), h.copy()), 0
        else:
            bad += 1
            if bad >= 4:
                break
    w, h = best_wh

    t0 = time.time()
    preds = []
    for i in range(0, nte, 4096):
        z = logits(pnp.array(Zte[i:i + 4096], requires_grad=False), w, h)
        preds.append((np.asarray(z) > 0).astype(np.uint8))
    pred = np.concatenate(preds)
    dt = time.time() - t0
    npar = int(w.size + h.size)
    return int((pred != yte).sum()), nte, dt, npar, ep + 1


# ============================================================================
# threshold estimation (validated on synthetic curves with a known crossing;
# that validation is methodology only and produces no reported result)
# ============================================================================
def _crossing(pa, ya, pb, yb):
    lp = np.log10(pa)
    da = np.log10(np.maximum(ya, 1e-12)) - np.log10(np.maximum(yb, 1e-12))
    sgn = np.sign(da)
    for i in range(len(lp) - 1):
        if sgn[i] == 0:
            return float(10 ** lp[i])
        if sgn[i] * sgn[i + 1] < 0:
            t = da[i] / (da[i] - da[i + 1])
            return float(10 ** (lp[i] + t * (lp[i + 1] - lp[i])))
    return None


def fit_thresholds(df):
    """(a) pairwise LER(p) curve crossings; (b) finite-size-scaling fit
    LER = A + Bx + Cx^2 with x = (p - p_th) d^(1/nu), restricted to the
    near-threshold window LER >= 0.01."""
    from scipy.optimize import least_squares

    out = {}
    ds = sorted(df.d.unique())
    for dec, g in df.groupby("decoder"):
        g = g.dropna(subset=["logical_error_rate"])
        if g["d"].nunique() < 2:
            continue
        rec = {"crossings": {}, "p_th_crossing": None, "p_th_fss": None,
               "nu": None, "fss_rss": None, "n_fss_points": 0,
               "bracketed": False, "larger_d_better": {}}
        cr = []
        for a, b in zip(ds[:-1], ds[1:]):
            ga = g[g.d == a].sort_values("p")
            gb = g[g.d == b].sort_values("p")
            if len(ga) != len(gb) or len(ga) < 2:
                continue
            c = _crossing(ga.p.to_numpy(), ga.logical_error_rate.to_numpy(),
                          gb.p.to_numpy(), gb.logical_error_rate.to_numpy())
            rec["crossings"][f"d{a}-d{b}"] = c
            if c is not None:
                cr.append(c)
        if cr:
            rec["p_th_crossing"] = float(np.mean(cr))
        for p in sorted(g.p.unique()):
            sub = g[g.p == p].sort_values("d")
            if len(sub) >= 2:
                v = sub.logical_error_rate.to_numpy()
                rec["larger_d_better"][f"{p:g}"] = bool(v[-1] < v[0])

        w = g[g.logical_error_rate >= 0.01]
        if w.d.nunique() >= 3 and len(w) >= 6:
            pv = w.p.to_numpy(float); dv = w.d.to_numpy(float)
            yv = w.logical_error_rate.to_numpy(float)

            def resid(th):
                p_th, nu, A, B, C = th
                x = (pv - p_th) * dv ** (1.0 / nu)
                return (A + B * x + C * x * x) - yv

            best = None
            p0s = ([rec["p_th_crossing"]] if rec["p_th_crossing"] else []) \
                + [0.02, 0.03, 0.05]
            for p0 in p0s:
                for nu0 in (0.8, 1.2, 2.0):
                    try:
                        r = least_squares(resid, [p0, nu0, 0.2, 1.0, 1.0],
                                          bounds=([1e-4, 0.3, -1, -1e5, -1e5],
                                                  [0.2, 4.0, 2, 1e5, 1e5]),
                                          max_nfev=20000)
                    except Exception:
                        continue
                    if best is None or r.cost < best.cost:
                        best = r
            if best is not None:
                rec["p_th_fss"] = float(best.x[0])
                rec["nu"] = float(best.x[1])
                rec["fss_rss"] = float(2 * best.cost)
                rec["n_fss_points"] = int(len(w))
        # A decoder whose LER rises with d at every sampled p has no threshold
        # in this window; the FSS fit is meaningless there, so suppress it.
        rec["no_threshold_in_window"] = bool(
            rec["p_th_crossing"] is None
            and len(rec["larger_d_better"]) > 0
            and not any(rec["larger_d_better"].values()))
        if rec["no_threshold_in_window"]:
            rec["p_th_fss"] = None
            rec["nu"] = None
            rec["fss_rss"] = None
        est = rec["p_th_crossing"] or rec["p_th_fss"]
        rec["bracketed"] = bool(est is not None and df.p.min() < est < df.p.max())
        out[dec] = rec
    return out


# ============================================================================
# figures
# ============================================================================
DEC_ORDER = ["MWPM", "Greedy", "MLP", "LogReg", "LogReg-PCA", "VQC-PCA"]
DEC_LABEL = {
    "MWPM": "MWPM (exact matching)",
    "Greedy": "Greedy matching",
    "MLP": "MLP (flat syndrome)",
    "LogReg": "Logistic reg. (flat syndrome)",
    "LogReg-PCA": f"Logistic reg., {N_PCA} PCA feat.",
    "VQC-PCA": f"VQC, {N_PCA} PCA feat. (quantum)",
}
DEC_COLOR = {"MWPM": "#111111", "Greedy": "#777777", "MLP": "#0072B2",
             "LogReg": "#E69F00", "LogReg-PCA": "#009E73", "VQC-PCA": "#CC79A7"}


def _panel_claim(g, decs):
    """A claim true of EVERY plotted series in the panel, or a description."""
    mono = all(
        len(s) >= 2 and np.all(np.diff(s) >= -1e-12)
        for s in (g[g.decoder == dec].sort_values("p")["logical_error_rate"]
                  .to_numpy() for dec in decs))
    mwpm_best = True
    for p in sorted(g.p.unique()):
        sub = g[g.p == p]
        if "MWPM" not in set(sub.decoder):
            mwpm_best = False
            break
        m = sub[sub.decoder == "MWPM"].logical_error_rate.iloc[0]
        if (sub.logical_error_rate < m - 1e-12).any():
            mwpm_best = False
            break
    if mono and mwpm_best:
        return "all rise with $p$; matching lowest"
    if mono:
        return "all rise with $p$"
    if mwpm_best:
        return "matching lowest at every $p$"
    return "logical error rate vs $p$"


def make_figures(df, thr):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300,
                         "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    ds = sorted(df.d.unique())
    decs = [x for x in DEC_ORDER if x in set(df.decoder)]
    ps = sorted(df.p.unique())

    def _pticks(ax):
        ax.set_xticks(ps)
        ax.set_xticklabels([f"{v:g}" for v in ps])
        ax.set_xticks([], minor=True)

    def _series(ax, s, dec, lw, ms):
        """Plot a series; zero-error points become 95% upper-bound markers."""
        y = s.logical_error_rate.to_numpy()
        nz = y > 0
        c = DEC_COLOR[dec]
        if nz.any():
            ax.plot(s.p.to_numpy()[nz], y[nz], "-o", color=c, lw=lw, ms=ms,
                    label=DEC_LABEL[dec], zorder=5 if dec == "MWPM" else 3)
            ax.fill_between(s.p.to_numpy()[nz], s.ler_ci_lo.to_numpy()[nz],
                            s.ler_ci_hi.to_numpy()[nz], color=c, alpha=0.15, lw=0)
        if (~nz).any():   # no logical errors observed -> plot the Wilson bound
            ax.plot(s.p.to_numpy()[~nz], s.ler_ci_hi.to_numpy()[~nz], "v",
                    color=c, mfc="white", ms=ms + 1.6, zorder=6,
                    label=None if nz.any() else DEC_LABEL[dec])

    # ---------------- Figure 1: threshold plot ----------------
    fig, axes = plt.subplots(1, len(ds), figsize=(3.15 * len(ds), 3.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        g = df[df.d == d]
        for dec in decs:
            s = g[g.decoder == dec].sort_values("p")
            if not s.empty:
                _series(ax, s, dec, 2.0 if dec == "MWPM" else 1.1, 3.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("physical error rate $p$")
        _pticks(ax)
        ax.set_title(f"$d$={d}: " + _panel_claim(g, decs), fontsize=7.5,
                     loc="left", pad=4)
        ax.grid(alpha=0.25, lw=0.4)
        ax.margins(0.07)
    axes[0].set_ylabel("logical error rate per shot")
    axes[0].text(0.04, 0.035, "lower = better", transform=axes[0].transAxes,
                 fontsize=6.5, style="italic", color="#444444")
    handles, labels = axes[-1].get_legend_handles_labels()
    if (df.logical_error_rate <= 0).any():
        handles.append(mpl.lines.Line2D([], [], marker="v", ls="none",
                                        color="#111111", mfc="white", ms=5))
        labels.append("no errors seen: 95% upper bound")
    axes[-1].legend(handles, labels, frameon=False, loc="lower right",
                    handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig("qec_threshold_fig.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 2: decoder comparison ----------------
    p_star = ps[1] if len(ps) > 1 else ps[0]     # well below threshold
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.3))

    gA = df[df.p == p_star]
    for dec in decs:
        s = gA[gA.decoder == dec].sort_values("d")
        if s.empty:
            continue
        y = s.logical_error_rate.to_numpy()
        nz = y > 0
        axA.plot(s.d.to_numpy()[nz], y[nz], "-o", color=DEC_COLOR[dec], lw=1.4,
                 ms=4, label=DEC_LABEL[dec])
        axA.fill_between(s.d.to_numpy()[nz], s.ler_ci_lo.to_numpy()[nz],
                         s.ler_ci_hi.to_numpy()[nz], color=DEC_COLOR[dec],
                         alpha=0.15, lw=0)
        if (~nz).any():
            axA.plot(s.d.to_numpy()[~nz], s.ler_ci_hi.to_numpy()[~nz], "v",
                     color=DEC_COLOR[dec], mfc="white", ms=5.6, zorder=6)
    axA.set_yscale("log"); axA.set_xticks(ds)
    axA.set_xlabel("code distance $d$")
    axA.set_ylabel("logical error rate per shot")
    # Title must be true of EVERY plotted series: classify each decoder's
    # behaviour across d at p_star, then pick a claim consistent with all of them.
    MATCHERS = {"MWPM", "Greedy"}
    improves, degrades = set(), set()
    for dec in decs:
        v = gA[gA.decoder == dec].sort_values("d").logical_error_rate.to_numpy()
        if len(v) != len(ds):
            continue
        if np.all(np.diff(v) <= 1e-12) and v[-1] < v[0]:
            improves.add(dec)
        if v[-1] > v[0]:
            degrades.add(dec)
    matchers_here = MATCHERS & set(decs)
    learned_here = set(decs) - MATCHERS
    if improves == matchers_here and degrades == learned_here:
        claim = "matching improves with distance;\nevery learned decoder degrades"
    elif improves == {"MWPM"}:
        claim = "only exact matching improves with distance"
    elif improves >= matchers_here and not (improves & learned_here):
        claim = "no learned decoder improves with distance"
    else:
        claim = "logical error rate versus code distance"
    axA.set_title(f"at $p$={p_star:g}, " + claim, fontsize=7.5, loc="left", pad=4)
    axA.grid(alpha=0.25, lw=0.4); axA.margins(0.10)
    axA.legend(frameon=False, loc="lower left", handlelength=1.6,
               borderpad=0.2, labelspacing=0.3)

    d_max = max(ds)
    gB = df[(df.d == d_max) & (df.p == p_star)].copy()
    gB = gB[gB.decoder.isin(decs)].sort_values("decode_sec_per_1k")
    # stagger labels so the three near-degenerate points stay legible
    offs = {"LogReg-PCA": (0, 9, "center"), "LogReg": (0, -14, "center"),
            "VQC-PCA": (-8, 9, "right"), "MLP": (0, 9, "center"),
            "Greedy": (0, -14, "center"), "MWPM": (-9, 2, "right")}
    for _, s in gB.iterrows():
        dec = s.decoder
        y = float(s.logical_error_rate)
        if y <= 0:
            y = float(s.ler_ci_hi)
        x = float(s.decode_sec_per_1k)
        axB.scatter(x, y, s=54, color=DEC_COLOR[dec], zorder=4,
                    edgecolor="white", lw=0.6)
        dx, dy, ha = offs.get(dec, (7, 3, "left"))
        axB.annotate(DEC_LABEL[dec], (x, y), textcoords="offset points",
                     xytext=(dx, dy), fontsize=6.5, color=DEC_COLOR[dec], ha=ha)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("decode time (s per 1000 shots)")
    axB.set_ylabel("logical error rate per shot")
    axB.set_title(f"accuracy against decode cost, $d$={d_max}, $p$={p_star:g}",
                  fontsize=7.5, loc="left", pad=4)
    axB.text(0.97, 0.06, "lower-left = better", transform=axB.transAxes,
             ha="right", va="bottom", fontsize=6.5, style="italic",
             color="#444444")
    axB.grid(alpha=0.25, lw=0.4)
    axB.set_xlim(2e-6, 4.0)
    axB.set_ylim(2e-5, 1.6)
    fig2.tight_layout()
    fig2.savefig("qec_decoder_comparison_fig.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("wrote qec_threshold_fig.png, qec_decoder_comparison_fig.png")


# ============================================================================
# main sweep
# ============================================================================
def load_config(d, p):
    z = np.load(os.path.join(DATA_DIR, f"cfg_d{d}_p{p}.npz"))
    n_det = int(z["n_detectors"])
    det = np.unpackbits(z["det_packed"], axis=1, count=n_det, bitorder="little")
    return det, z["obs"].astype(np.uint8), n_det


def main():
    with open(os.path.join(DATA_DIR, "manifest.json")) as fh:
        man = json.load(fh)
    n_tr_pool = man["n_train_shots"]

    val = validate_matching()
    with open("qec_matching_validation.json", "w") as fh:
        json.dump(val, fh, indent=2)
    print(f"MWPM validation: {val['mwpm_matches_bruteforce']}/{val['n_trials']} "
          f"instances match brute force; greedy optimal on "
          f"{val['greedy_optimal_fraction']:.1%}", flush=True)

    from sklearn.decomposition import PCA

    # resume: keep rows already computed for (decoder, d, p)
    rows = []
    done = set()
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        rows = prev.to_dict("records")
        done = {(r["decoder"], r["d"], r["p"]) for r in rows}
        print(f"resuming: {len(rows)} rows already in {RESULTS_CSV}")

    for cfg in man["configs"]:
        d, p = cfg["d"], cfg["p"]
        if ("MWPM", d, p) in done:
            continue
        det, obs, n_det = load_config(d, p)
        n_stab = d - 1
        det_tr, det_te = det[:n_tr_pool], det[n_tr_pool:]
        y_tr, y_te = obs[:n_tr_pool], obs[n_tr_pool:]
        base = dict(d=d, p=p, n_detectors=n_det)
        print(f"\n=== d={d} p={p} ({n_det} detectors, "
              f"{cfg['mean_defects_per_shot']:.1f} defects/shot) ===", flush=True)

        def add(name, kind, res, n_train, extra=""):
            errs, n, dt, npar = res[0], res[1], res[2], res[3]
            lo, hi = wilson_ci(errs, n)
            rows.append(dict(base, decoder=name, kind=kind, n_train_shots=n_train,
                             n_test_shots=n, logical_error_rate=errs / n,
                             ler_ci_lo=lo, ler_ci_hi=hi,
                             decode_sec_per_1k=1000.0 * dt / n, n_params=npar,
                             n_logical_errors=errs))
            print(f"  {name:9s} LER={errs/n:.4e} [{lo:.2e},{hi:.2e}] "
                  f"n_test={n} {1000*dt/n:.4f} s/1k params={npar} {extra}",
                  flush=True)

        add("MWPM", "classical baseline",
            run_matching(det_te, y_te, n_stab, d, mwpm_predict, MWPM_TIME_BUDGET), 0)
        add("Greedy", "classical baseline",
            run_matching(det_te, y_te, n_stab, d, greedy_predict,
                         GREEDY_TIME_BUDGET), 0)
        add("LogReg", "neural", run_logreg(det_tr, y_tr, det_te, y_te), len(det_tr))
        add("MLP", "neural", run_mlp(det_tr, y_tr, det_te, y_te),
            min(MLP_MAX_TRAIN, len(det_tr)))

        pca = PCA(n_components=N_PCA, random_state=SEED)
        pca.fit(det_tr[:100_000].astype(np.float32))
        Z = pca.transform(det_tr[:VQC_MAX_TRAIN])
        pca_state = (pca, Z.mean(0), Z.std(0) + 1e-9)
        evr = float(pca.explained_variance_ratio_.sum())
        add("LogReg-PCA", "classical baseline",
            run_logreg_pca(det_tr, y_tr, det_te, y_te, pca_state), len(det_tr),
            extra=f"(control on the VQC's own features, EVR={evr:.3f})")
        add("VQC-PCA", "hybrid quantum-classical",
            run_vqc(det_tr, y_tr, det_te, y_te, pca_state),
            min(VQC_MAX_TRAIN, len(det_tr)), extra=f"(PCA EVR={evr:.3f})")

        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(rows)
    cols = ["decoder", "kind", "d", "p", "n_train_shots", "n_test_shots",
            "logical_error_rate", "ler_ci_lo", "ler_ci_hi", "decode_sec_per_1k",
            "n_params", "n_logical_errors", "n_detectors"]
    df[cols].to_csv(RESULTS_CSV, index=False)
    print(f"\nwrote {RESULTS_CSV}: {len(df)} rows")

    thr = fit_thresholds(df)
    with open("qec_thresholds.json", "w") as fh:
        json.dump(thr, fh, indent=2)
    make_figures(df, thr)


if __name__ == "__main__":
    main()
