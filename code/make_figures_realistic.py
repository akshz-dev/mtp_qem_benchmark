"""Figures for the device-realistic regime (Chapter 5, regime II).

Reads sweep_realistic.csv and writes:
  fig_ch5_scaling.png   parameter scaling vs unitary folding, both splits
  fig_ch5_regime.png    memory: production vs reset (memoryless) and defect-free controls
  fig_ch5_robustness.png ranking over 10 grouped splits; VQC seed spread
  fig_ch5_ood_features.png depth extrapolation with and without structural features
  fig_ch5_drift.png     calibration drift
  fig_ch5_shotkernel.png shot-limited quantum kernel: error vs circuit executions

Usage: python make_figures_realistic.py --outdir <dir>
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAME = {"no_mitigation": "No mitigation", "zne_linear": "ZNE, linear",
        "zne_richardson2": "ZNE, Richardson (2)", "zne_exponential": "ZNE, exponential",
        "ridge": "Ridge regression", "hist_gbm": "Gradient-boosted trees",
        "mlp": "Neural net (64,64)", "qk_ridge_quantum": "Quantum kernel ridge",
        "dr_vqc_quantum": "Data-reuploading VQC",
        "bounded_vqc_quantum": "Bounded-encoding VQC"}
KIND_C = {"unmitigated": "0.45", "analytic ZNE": "#6f8faf",
          "classical ML": "#3f7f5f", "hybrid quantum-classical": "#c2591a"}
ORDER = ["no_mitigation", "zne_exponential", "zne_richardson2", "zne_linear",
         "dr_vqc_quantum", "bounded_vqc_quantum", "ridge", "hist_gbm", "mlp",
         "qk_ridge_quantum"]


def style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "figure.facecolor": "white"})


def letter(ax, s, dx=-0.14):
    ax.text(dx, 1.05, s, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left")


def dumbbell(ax, d, cfgs, labels, colors, title, xlabel="RMSE vs ideal"):
    """One row per model, one marker per configuration."""
    present = [m for m in ORDER if m in set(d["model"])]
    y = np.arange(len(present))
    for cfg, lab, col in zip(cfgs, labels, colors):
        s = d[d["config"] == cfg].set_index("model")
        xs = [s.loc[m, "rmse"] if m in s.index else np.nan for m in present]
        ax.plot(xs, y, "o", color=col, ms=6, mec="white", mew=0.7, label=lab, zorder=3)
    for yi, m in zip(y, present):
        vals = [d[(d.config == c) & (d.model == m)]["rmse"] for c in cfgs]
        vals = [float(v.iloc[0]) for v in vals if len(v)]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [yi, yi], color="0.75", lw=1.2, zorder=1)
    base = d[(d.model == "no_mitigation") & (d.config == cfgs[0])]["rmse"]
    if len(base):
        ax.axvline(float(base.iloc[0]), color="0.5", ls=":", lw=1.0, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in present])
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    ax.margins(y=0.06)
    ax.grid(axis="y", alpha=0)


def fig_scaling(sw, out):
    d = sw[sw.experiment == "R1_scaling_method"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, sp, t in [(axes[0], "A_random_circuits", "Held-out random circuits"),
                      (axes[1], "B_depth_extrapolation", "Deeper circuits than trained on")]:
        dumbbell(ax, d[d.split == sp], ["parameter_scaling", "unitary_folding"],
                 ["parameter scaling (idealized)", "unitary folding (hardware-realizable)"],
                 ["#6f8faf", "#c2591a"], t)
    lo = d["rmse"].min(); hi = min(d["rmse"].max() * 1.08, 1.0)
    axes[0].set_xlim(0, hi)
    axes[0].legend(loc="upper right")
    for a, l in zip(axes, "ab"):
        letter(a, l, dx=-0.42 if l == "a" else -0.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_scaling.png"), bbox_inches="tight")
    plt.close(fig)


def fig_regime(sw, out):
    d = sw[sw.experiment == "R2_markovianity"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, scal, t in [(axes[0], "parameter_scaling", "Parameter scaling"),
                        (axes[1], "unitary_folding", "Unitary folding (hardware-realizable)")]:
        e = d[d.config.str.startswith(scal + ":")].copy()
        e["config"] = e["config"].str.split(":").str[1]
        dumbbell(ax, e, ["markovian_control", "reset_control", "nonmarkovian_production"],
                 ["defect removed", "defect reset every layer (no memory)",
                  "defect with memory (production)"],
                 ["#3f7f5f", "#d79a1e", "#c2591a"], t)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.03))
    for a, l in zip(axes, "ab"):
        letter(a, l, dx=-0.42 if l == "a" else -0.05)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "fig_ch5_regime.png"), bbox_inches="tight")
    plt.close(fig)


def fig_drift(sw, out):
    d = sw[sw.experiment == "R3_calibration_drift"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    dumbbell(ax, d, ["no_drift_heldout", "drift_training_circuits", "drift_heldout"],
             ["calibration A, held-out circuits", "calibration B, training circuits",
              "calibration B, held-out circuits"],
             ["#3f7f5f", "#d79a1e", "#c2591a"],
             "After a 30% recalibration, ridge is the only learned arm\nwhose held-out error rises resolvably")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_drift.png"), bbox_inches="tight")
    plt.close(fig)


def fig_shotkernel(sw, out):
    d = sw[sw.experiment == "R4_shot_limited_kernel"].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    fin = (d[d.shots_per_entry > 0].groupby("shots_per_entry")
           .agg(execs=("circuit_executions", "first"), rmse=("rmse", "mean"),
                lo=("rmse", "min"), hi=("rmse", "max"), n=("rmse", "size")).reset_index())
    ex = d[d.shots_per_entry == 0]
    ax.errorbar(fin["execs"], fin["rmse"], yerr=[fin["rmse"] - fin["lo"], fin["hi"] - fin["rmse"]],
                fmt="-o", color="#c2591a", ms=6, lw=1.4, capsize=2,
                label=f"shot-estimated Gram, mean and range of {int(fin['n'].iloc[0])} draws")
    import matplotlib.transforms as mtr
    tr_ = mtr.blended_transform_factory(ax.transData, ax.transAxes)
    for _, r in fin.iterrows():
        ax.text(r.execs, 1.02, f"$10^{{{int(np.log10(r.shots_per_entry))}}}$", transform=tr_,
                ha="center", va="bottom", fontsize=7, color="0.35")
    ax.text(1.0, 1.08, "shots per Gram-matrix entry", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, color="0.35")
    if len(ex):
        ax.axhline(float(ex["rmse"].iloc[0]), color="#c2591a", ls="--", lw=1.0,
                   label=f"exact Gram (simulator only): {float(ex['rmse'].iloc[0]):.3f}")
    base = float(d["unmitigated_rmse"].iloc[0])
    ax.axhline(base, color="0.45", ls=":", lw=1.1, label=f"no mitigation: {base:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Total circuit executions to build and apply the kernel")
    ax.set_ylabel("RMSE vs ideal")
    ax.set_title("The shot-estimated kernel needs $10^5$ shots per entry\nto come close to its exact-Gram accuracy", loc="left", pad=24)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_shotkernel.png"), bbox_inches="tight")
    plt.close(fig)
    return fin, ex, base


def fig_robustness(out):
    """(a) mean +- SD of test RMSE over 10 circuit-grouped splits, three settings;
    (b) variational arms over 10 initialisation seeds at the fixed seed-7 split."""
    cfgs = [("ideal", "regime I", "#6f8faf"), ("real_param", "regime II, parameter scaling", "#3f7f5f"),
            ("real_fold", "regime II, unitary folding", "#c2591a")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    present = ORDER
    y = np.arange(len(present))
    for k, (c, lab, col) in enumerate(cfgs):
        r = pd.read_csv(f"robust_splits_{c}.csv").groupby("model")["rmse"].agg(["mean", "std"])
        ax.errorbar([r.loc[m, "mean"] for m in present], y + (k - 1) * 0.22,
                    xerr=[r.loc[m, "std"] for m in present], fmt="o", color=col, ms=4.5,
                    mec="white", mew=0.5, elinewidth=1.0, capsize=1.5, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in present])
    ax.set_xlabel("Test RMSE, mean $\\pm$ SD over 10 circuit-grouped splits")
    ax.set_title("Ranking over 10 splits", loc="left")
    ax.grid(axis="y", alpha=0); ax.legend(loc="lower right")
    ax = axes[1]
    v = pd.read_csv("vqc_seed_sweep.csv")
    v = v[v.epochs == 40]
    rows = [("A_random_circuits", "dr_vqc_quantum", "Reuploading, Split A"),
            ("A_random_circuits", "bounded_vqc_quantum", "Bounded, Split A"),
            ("B_depth_extrapolation", "dr_vqc_quantum", "Reuploading, Split B"),
            ("B_depth_extrapolation", "bounded_vqc_quantum", "Bounded, Split B")]
    base = pd.read_csv("benchmark_results.csv") if os.path.exists("benchmark_results.csv") else None
    for i, (sp, m, lab) in enumerate(rows):
        e = v[(v.split == sp) & (v.model == m)]
        jit = (np.arange(len(e)) - len(e) / 2) * 0.025
        ax.plot(e["rmse"], i + jit, "o", color=KIND_C["hybrid quantum-classical"], ms=4.5,
                mec="white", mew=0.5, alpha=0.9)
        s7 = e[e.seed == 7]["rmse"]
        if len(s7):
            ax.plot(float(s7.iloc[0]), i, "D", color="black", ms=4.5, mfc="none", mew=0.9,
                    label="seed 7 (reported in v01)" if i == 0 else None)
        if base is not None:
            u = base[(base.split == sp) & (base.model == "no_mitigation")]["rmse"]
            if len(u):
                ax.plot([float(u.iloc[0])] * 2, [i - 0.3, i + 0.3], color="0.45", ls=":", lw=1.1,
                        label="no mitigation" if i == 0 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows])
    ax.set_xlabel("Test RMSE, one point per initialisation seed")
    ax.set_title("Variational arms over 10 seeds", loc="left")
    ax.grid(axis="y", alpha=0); ax.legend(loc="lower right")
    for a, l in zip(axes, "ab"):
        letter(a, l, dx=-0.42 if l == "a" else -0.36)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_robustness.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ood_features(out):
    fr = pd.read_csv("followup_results.csv")
    d = fr[fr.experiment == "F1_ood_features"]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    arms = [m for m in ORDER if m in set(d.model) and m != "no_mitigation"]
    y = np.arange(len(arms))
    for k, (c, lab, col) in enumerate([("full", "all six features (depth, $n_{2q}$ included)", "0.55"),
                                       ("noise_only", "three noise-scaled values only", "#c2591a")]):
        e = d[d.config == c].set_index("model")
        ax.errorbar([e.loc[m, "rmse"] for m in arms], y + (k - 0.5) * 0.24,
                    xerr=[[e.loc[m, "rmse"] - e.loc[m, "rmse_lo"] for m in arms],
                          [e.loc[m, "rmse_hi"] - e.loc[m, "rmse"] for m in arms]],
                    fmt="o", color=col, ms=5, mec="white", mew=0.5, capsize=1.5, label=lab)
    u = float(d[d.model == "no_mitigation"]["rmse"].iloc[0])
    ax.axvline(u, color="0.45", ls=":", lw=1.1, label=f"no mitigation: {u:.3f}")
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in arms])
    ax.set_xlabel("Test RMSE on deeper circuits (Split B), log scale")
    ax.set_title("Without the out-of-range structural features, every learned arm\nbeats no mitigation on deeper circuits", loc="left")
    ax.grid(axis="y", alpha=0); ax.legend(loc="upper right", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_ood_features.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    style()
    sw = pd.read_csv("sweep_realistic.csv")
    fig_scaling(sw, a.outdir)
    fig_regime(sw, a.outdir)
    fig_drift(sw, a.outdir)
    fig_shotkernel(sw, a.outdir)
    fig_robustness(a.outdir)
    fig_ood_features(a.outdir)
    print("wrote:", sorted(f for f in os.listdir(a.outdir)
                           if f.startswith("fig_ch5_") and f.endswith(".png")))


if __name__ == "__main__":
    main()
