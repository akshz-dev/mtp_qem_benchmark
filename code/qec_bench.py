"""
qec_bench.py -- benchmark classical matching decoders against ML syndrome
decoders on the repetition-code / phenomenological-noise data produced by
qec_gen_data.py.

SCOPE NOTE. The benchmark uses the repetition code under phenomenological noise,
not the surface code under circuit-level noise (thesis Section 6.1). Every number
produced here comes from the numpy simulation in qec_gen_data.py.

Decoders: exact and greedy matching, exact logical-class posterior decoding,
logistic regression, an MLP, PCA logistic regression, and a PCA-based VQC.

Reports held-out logical error rates and Wilson intervals, a common test prefix
for ranking, and paired McNemar tests on each pair's shared shots.

Run:  python qec_bench.py
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

DATA_DIR = "qec_data"
RESULTS_CSV = "qec_decoder_results.csv"
SEED = 7

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
# ML decoders
# ============================================================================
def run_logreg(det_tr, y_tr, det_te):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(det_tr, y_tr)
    t0 = time.perf_counter()
    pred = clf.predict(det_te)
    return pred, time.perf_counter() - t0, int(clf.coef_.size + clf.intercept_.size)


def run_logreg_pca(det_tr, y_tr, det_te, pca_state):
    """Logistic regression on the six PCA scores, without the VQC's angle transform."""
    from sklearn.linear_model import LogisticRegression

    pca = pca_state[0]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(pca.transform(det_tr), y_tr)
    t0 = time.perf_counter()
    pred = clf.predict(pca.transform(det_te))
    return pred, time.perf_counter() - t0, int(clf.coef_.size + clf.intercept_.size)


def run_mlp(det_tr, y_tr, det_te):
    from sklearn.neural_network import MLPClassifier

    n = min(MLP_MAX_TRAIN, len(det_tr))
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation="relu",
                        alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                        max_iter=MLP_MAX_ITER, early_stopping=True,
                        n_iter_no_change=MLP_NO_CHANGE,
                        validation_fraction=0.1, random_state=SEED)
    clf.fit(det_tr[:n], y_tr[:n])
    t0 = time.perf_counter()
    pred = clf.predict(det_te)
    dt = time.perf_counter() - t0
    npar = sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    return pred, dt, int(npar)


def run_vqc(det_tr, y_tr, det_te, pca_state):
    """Data-reuploading VQC on N_PCA PCA features with a logistic head."""
    import pennylane as qml
    from pennylane import numpy as pnp

    pca, mu, sd = pca_state
    ntr = min(VQC_MAX_TRAIN, len(det_tr))
    nte = min(VQC_MAX_TEST, len(det_te))
    Ztr = np.clip((pca.transform(det_tr[:ntr]) - mu) / sd, -3, 3) * (np.pi / 3)
    ytr = y_tr[:ntr].astype(np.float64)

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

    t0 = time.perf_counter()
    Zte = np.clip((pca.transform(det_te[:nte]) - mu) / sd, -3, 3) * (np.pi / 3)
    preds = []
    for i in range(0, nte, 4096):
        z = logits(pnp.array(Zte[i:i + 4096], requires_grad=False), w, h)
        preds.append((np.asarray(z) > 0).astype(np.uint8))
    pred = np.concatenate(preds)
    return pred, time.perf_counter() - t0, int(w.size + h.size)


# ============================================================================
# descriptive finite-distance crossover diagnostics
# ============================================================================
def _crossing(p, ya, yb):
    # Equal point estimates do not establish a crossing. In particular,
    # observing zero errors in both samples gives only upper limits.
    ordered = (ya > 0) & (yb > 0) & (ya != yb)
    lp = np.log10(p[ordered])
    delta = np.log10(ya[ordered]) - np.log10(yb[ordered])
    for i in np.flatnonzero(delta[:-1] * delta[1:] < 0):
        t = delta[i] / (delta[i] - delta[i + 1])
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
            if len(ga) < 2 or not np.array_equal(ga.p.to_numpy(), gb.p.to_numpy()):
                continue
            c = _crossing(ga.p.to_numpy(), ga.logical_error_rate.to_numpy(),
                          gb.logical_error_rate.to_numpy())
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
        # Suppress the descriptive fit when neither crossings nor endpoint
        # improvement support one. Retain the historical artifact field name.
        rec["no_threshold_in_window"] = bool(
            rec["p_th_crossing"] is None
            and len(rec["larger_d_better"]) > 0
            and not any(rec["larger_d_better"].values()))
        if rec["no_threshold_in_window"]:
            rec["p_th_fss"] = None
            rec["nu"] = None
            rec["fss_rss"] = None
        rec["bracketed"] = rec["p_th_crossing"] is not None
        out[dec] = rec
    return out


# ============================================================================
# figures
# ============================================================================
DEC_STYLES = {
    "Exact-MAP": ("Exact logical MAP", "#007a59", "-"),
    "MWPM": ("MWPM (PyMatching)", "#171717", "--"),
    "Greedy": ("Greedy matching", "#7d7d7d", ":"),
    "MLP": ("MLP", "#0072b2", "-"),
    "LogReg": ("Logistic regression", "#bd5900", ":"),
    "LogReg-PCA": ("PCA logistic regression", "#6550a4", "--"),
    "VQC-PCA": ("PCA + VQC", "#b02b76", "-."),
}


def plot_ler(ax, data, x_col):
    """Plot each decoder on its available shots, with zero counts as limits."""
    for decoder, (label, color, style) in DEC_STYLES.items():
        s = data[data.decoder == decoder].sort_values(x_col)
        x = s[x_col].to_numpy()
        y = s.logical_error_rate.to_numpy()
        positive = y > 0
        lo, hi = s.ler_ci_lo.to_numpy(), s.ler_ci_hi.to_numpy()
        if positive.any():
            ax.plot(x, np.where(positive, y, np.nan), color=color, ls=style,
                    marker="o", ms=2.8, lw=1.4, label=label)
            ax.fill_between(x, lo, hi, where=positive, color=color, alpha=0.12, lw=0)
        if (~positive).any():
            ax.plot(x[~positive], hi[~positive], color=color, marker="v",
                    ls="none", mfc="white", ms=5, label=None if positive.any() else label)
    ax.set_yscale("log")
    ax.grid(which="major", alpha=0.24, lw=0.5)


def make_figures(df, matching_validation):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 300,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    ds = sorted(df.d.unique())
    ps = sorted(df.p.unique())

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.4), sharey=True,
                             constrained_layout=True)
    for ax, d in zip(axes.flat, ds):
        plot_ler(ax, df[df.d == d], "p")
        ax.set_xscale("log")
        ax.set_xticks(ps)
        ax.set_xticklabels([f"{p:g}" if p in (0.01, 0.02, 0.04, 0.08, 0.15) else "" for p in ps])
        ax.set_xticks([], minor=True)
        ax.set_title(f"Distance {d}", loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Logical error rate per shot")
    for ax in axes[-1]:
        ax.set_xlabel("Physical error rate $p$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if (df.logical_error_rate == 0).any():
        handles.append(mpl.lines.Line2D([], [], marker="v", ls="none", color="#171717",
                                      mfc="white", ms=5))
        labels.append("Zero observed errors (95% upper limit)")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig("qec_threshold_fig.png", dpi=300)
    plt.close(fig)

    p_star = 0.02
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.9), constrained_layout=True)
    plot_ler(left, df[df.p == p_star], "d")
    left.set_xticks(ds)
    left.set_xlabel("Code distance $d$")
    left.set_ylabel("Logical error rate per shot")
    left.set_title(r"a  Distance scaling at $p=0.02$", loc="left")
    left.legend(loc="lower left", frameon=False, fontsize=7.2)

    risks = matching_validation["analytic_logical_error_rate_by_p"]
    ps = sorted(map(float, risks))
    rows = [risks[f"{p:g}"] for p in ps]
    map_risk = np.array([row["logical_class_map"] for row in rows])

    def excess(key):
        return 100 * np.maximum(0, np.array([row[key] for row in rows]) - map_risk)

    right.fill_between(ps, excess("min_across_mwpm_tie_choices"),
                       excess("max_across_mwpm_tie_choices"), color="#0072b2", alpha=0.2,
                       label="Range over valid MWPM tie choices")
    right.plot(ps, excess("pymatching"), "--o", color="#171717", lw=1.4, ms=3,
               label="PyMatching tie choice")
    right.axhline(0, color="#007a59", lw=1.2, label="Exact logical MAP")
    right.set_xticks([0.01, 0.05, 0.10, 0.15])
    right.set_xlabel("Physical error rate $p$")
    right.set_ylabel("Excess logical error (percentage points)")
    right.set_title(r"b  Exact tie sensitivity at $d=3$", loc="left")
    right.grid(alpha=0.24, lw=0.5)
    right.legend(loc="upper left", frameon=False, fontsize=7.2)
    fig.savefig("qec_decoder_comparison_fig.png", dpi=300)
    plt.close(fig)

# ============================================================================
# main sweep
# ============================================================================
def load_config(data_dir, d, p):
    path = data_dir / f"cfg_d{d}_p{p}.npz"
    with np.load(path, allow_pickle=False) as archive:
        n_det = int(archive["n_detectors"])
        packed, obs = archive["det_packed"], archive["obs"]
        if (n_det != (d + 1) * (d - 1)
                or obs.ndim != 1 or not np.isin(obs, (0, 1)).all()
                or packed.shape != (len(obs), (n_det + 7) // 8) or packed.dtype != np.uint8):
            raise ValueError(f"Invalid detector or logical-label arrays: {path}")
        for field, expected in (("d", d), ("p", p), ("rounds", d)):
            if field in archive and archive[field].item() != expected:
                raise ValueError(f"Archive {field} differs from the manifest: {path}")
        det = np.unpackbits(packed, axis=1, count=n_det, bitorder="little")
        return det, obs.astype(np.uint8)


def training_seen_mask(det_tr, det_te):
    """Identify test syndromes recorded at least once in the training block."""
    def packed_rows(det):
        packed = np.packbits(det, axis=1, bitorder="little")
        return packed.view(np.dtype((np.void, packed.shape[1]))).ravel()

    observed = np.unique(packed_rows(det_tr))
    return len(observed), np.isin(packed_rows(det_te), observed)


def paired_errors(model_errors, ref_errors):
    from scipy.stats import binomtest

    n = min(len(model_errors), len(ref_errors))
    model, ref = model_errors[:n], ref_errors[:n]
    model_only = int(np.count_nonzero(model & ~ref))
    ref_only = int(np.count_nonzero(ref & ~model))
    discordant = model_only + ref_only
    return dict(n_paired_shots=n, model_only_errors=model_only, ref_only_errors=ref_only,
                ler_difference=(model_only - ref_only) / n,
                mcnemar_p=binomtest(model_only, discordant, 0.5).pvalue if discordant else 1.0)


def benchmark_config(data_dir, cfg, n_tr_pool, n_te_pool=None):
    from sklearn.decomposition import PCA
    from qec_matching import decode_greedy, decode_mwpm
    from qec_posterior import ExactLogicalDecoder

    d, p = cfg["d"], cfg["p"]
    det, obs = load_config(data_dir, d, p)
    if not 0 < n_tr_pool < len(obs) or (n_te_pool is not None and len(obs) != n_tr_pool + n_te_pool):
        raise ValueError(f"Shot count differs from the manifest for d={d}, p={p:g}")
    n_det = det.shape[1]
    det_tr, det_te = det[:n_tr_pool], det[n_tr_pool:]
    y_tr, y_te = obs[:n_tr_pool], obs[n_tr_pool:]
    distinct_train, seen = training_seen_mask(det_tr, det_te)
    n_common = min(VQC_MAX_TEST, len(y_te))
    base = dict(d=d, p=p)
    rows, errors, conditional = [], {}, {}

    def add(name, kind, result, n_train):
        pred, dt, npar = result
        n = len(pred)
        wrong = pred != y_te[:n]
        errors[name] = wrong
        errs = int(wrong.sum())
        common_errs = int(wrong[:n_common].sum())
        lo, hi = wilson_ci(errs, n)
        rows.append(dict(decoder=name, kind=kind, **base,
                         n_train_shots=n_train, n_test_shots=n,
                         logical_error_rate=errs / n, ler_ci_lo=lo, ler_ci_hi=hi,
                         decode_sec_per_1k=1000.0 * dt / n,
                         n_params=npar, n_logical_errors=errs, n_detectors=n_det,
                         n_common_test_shots=n_common, n_common_logical_errors=common_errs,
                         common_logical_error_rate=common_errs / n_common))
        mask = seen[:n]
        n_seen = int(mask.sum())
        conditional[name] = {
            "n_seen": n_seen, "n_unseen": n - n_seen,
            "n_errors_seen": int(wrong[mask].sum()),
            "n_errors_unseen": int(wrong[~mask].sum()),
        }

    def timed_decode(decode, *args):
        t0 = time.perf_counter()
        pred = decode(*args)
        return pred, time.perf_counter() - t0, 0

    add("MWPM", "classical baseline", timed_decode(decode_mwpm, det_te, d), 0)
    add("Greedy", "classical baseline", timed_decode(decode_greedy, det_te, d), 0)
    oracle = ExactLogicalDecoder(d, p)
    add("Exact-MAP", "model-informed control", timed_decode(oracle.predict, det_te), 0)
    add("LogReg", "classical learned", run_logreg(det_tr, y_tr, det_te), len(det_tr))
    add("MLP", "classical learned", run_mlp(det_tr, y_tr, det_te),
        min(MLP_MAX_TRAIN, len(det_tr)))

    pca = PCA(n_components=N_PCA, random_state=SEED)
    pca.fit(det_tr[:100_000].astype(np.float32))
    scores = pca.transform(det_tr[:VQC_MAX_TRAIN])
    pca_state = pca, scores.mean(0), scores.std(0) + 1e-9
    add("LogReg-PCA", "classical learned", run_logreg_pca(det_tr, y_tr, det_te, pca_state),
        len(det_tr))
    add("VQC-PCA", "hybrid quantum-classical", run_vqc(det_tr, y_tr, det_te, pca_state),
        min(VQC_MAX_TRAIN, len(det_tr)))

    pairs = [dict(**base, decoder=name, reference="MWPM", **paired_errors(wrong, errors["MWPM"]))
             for name, wrong in errors.items() if name != "MWPM"]
    pairs.append(dict(**base, decoder="MLP", reference="Exact-MAP",
                      **paired_errors(errors["MLP"], errors["Exact-MAP"])))
    coverage = dict(**base, n_detectors=n_det, possible_syndromes=1 << n_det,
                    n_train_shots=len(det_tr), distinct_train=distinct_train,
                    n_test_shots=len(det_te), n_test_seen=int(seen.sum()),
                    test_seen_fraction=float(seen.mean()), decoder_counts=conditional)
    pca_evr = dict(**base, evr6=float(pca.explained_variance_ratio_.sum()))
    return rows, pairs, coverage, pca_evr


def run_config(job):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return benchmark_config(*job)


def publish_results(completed):
    rows, pairs, coverage, evr = [], [], {}, {}
    for config_rows, config_pairs, config_coverage, config_evr in completed:
        rows.extend(config_rows)
        pairs.extend(config_pairs)
        key = f"d{config_coverage['d']}_p{config_coverage['p']:g}"
        coverage[key] = config_coverage
        evr[key] = config_evr

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    paired = pd.DataFrame(pairs)
    raw_p = paired.mcnemar_p.to_numpy()
    order = np.argsort(raw_p)
    adjusted = np.empty_like(raw_p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(raw_p[order] * np.arange(len(order), 0, -1)))
    paired["mcnemar_p_holm"] = adjusted
    paired.to_csv("qec_paired_results.csv", index=False)
    for path, data in (("qec_syndrome_coverage.json", coverage), ("qec_pca_evr.json", evr),
                       ("qec_thresholds.json", fit_thresholds(df))):
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"wrote {len(df)} decoder results and {len(paired)} paired comparisons", flush=True)
    return df


def main():
    from qec_matching import validate_matching
    from qec_posterior import validate_exact_posterior

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="parallel configurations")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    data_dir = Path(DATA_DIR).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    n_train = manifest["n_train_shots"]
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scipy", "scikit-learn", "pennylane", "matplotlib", "pymatching")}
    provenance = dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                      workers=args.workers, threadpool_limit=1, master_seed=manifest["master_seed"],
                      n_train_pool=n_train, n_test_pool=manifest["n_test_shots"],
                      common_test_prefix=min(VQC_MAX_TEST, manifest["n_test_shots"]))
    matching_validation = validate_matching()
    posterior_validation = validate_exact_posterior()

    jobs = [(data_dir, cfg, n_train, manifest["n_test_shots"]) for cfg in manifest["configs"]]
    completed = []
    if args.workers == 1:
        for job in jobs:
            completed.append(run_config(job))
            print(f"completed d={job[1]['d']} p={job[1]['p']:g}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_config, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                completed.append(future.result())
                print(f"completed d={job[1]['d']} p={job[1]['p']:g}", flush=True)
    completed.sort(key=lambda result: (result[2]["d"], result[2]["p"]))
    df = publish_results(completed)
    make_figures(df, matching_validation)
    for path, data in (("qec_matching_validation.json", matching_validation),
                       ("qec_posterior_validation.json", posterior_validation),
                       ("qec_run_metadata.json", provenance)):
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
