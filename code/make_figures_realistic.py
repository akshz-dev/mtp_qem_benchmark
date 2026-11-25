"""Figures for the device-informed regime (Chapter 6).

Reads sweep_realistic.csv and the other files in results/ and writes:
  fig_ch5_nonmarkov.png idle trace distance and coupling sweep (nonmarkovianity_check.json)
  fig_ch5_scaling.png   parameter scaling vs unitary folding, both splits
  fig_ch5_regime.png    memory: production vs reset (memoryless) and defect-free controls
  fig_ch5_robustness.png ranking over 10 grouped splits; VQC seed spread
  fig_ch5_ood_features.png depth extrapolation with and without structural features
  fig_ch5_drift.png     calibration drift
  fig_ch5_shotkernel.png shot-limited quantum kernel: error vs tuning and final execution costs

Use --folding-seed-pairs for the circuit-pool and shot-draw replication.

Usage: python make_figures_realistic.py --outdir <dir>
"""
import argparse
from itertools import cycle
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

NAME = {"no_mitigation": "No mitigation", "zne_linear": "ZNE, linear",
        "zne_richardson2": "ZNE, Richardson (2)", "zne_exponential": "ZNE, exponential",
        "ridge": "Ridge regression", "hist_gbm": "Gradient-boosted trees",
        "mlp": "Neural net (64,64)", "rbf_krr": "Classical RBF kernel ridge",
        "qk_ridge_quantum": "Quantum kernel ridge",
        "dr_vqc_quantum": "Data-reuploading VQC",
        "bounded_vqc_quantum": "Bounded-encoding VQC"}
KIND_C = {"unmitigated": "0.45", "analytic ZNE": "#6f8faf",
          "classical ML": "#3f7f5f", "hybrid quantum-classical": "#c2591a"}
ORDER = ["no_mitigation", "zne_exponential", "zne_richardson2", "zne_linear",
         "dr_vqc_quantum", "bounded_vqc_quantum", "ridge", "hist_gbm", "mlp",
         "rbf_krr", "qk_ridge_quantum"]
SEED_SPLITS = [("A_random_circuits", "A. Random circuits"),
               ("B_depth_extrapolation", "B. Deeper circuits")]
SEED_PROTOCOLS = [("parameter_scaling", "Parameter scaling"),
                  ("unitary_folding", "Unitary folding")]
POOL_COLORS = ["#0072b2", "#d55e00", "#00845f"]
SHOT_MARKERS = ["o", "s", "^"]


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


def dumbbell(ax, d, cfgs, labels, colors, title, xlabel="RMSE vs ideal", shared_baseline=False):
    """One row per model, one marker per configuration."""
    present = [m for m in ORDER if m in set(d["model"])]
    y = np.arange(len(present))
    methods = [m for m in present if not shared_baseline or m != "no_mitigation"]
    positions = [present.index(m) for m in methods]
    for cfg, lab, col in zip(cfgs, labels, colors):
        s = d[d["config"] == cfg].set_index("model")
        ax.plot(s["rmse"].reindex(methods), positions, "o", color=col, ms=6,
                mec="white", mew=0.7, label=lab, zorder=3)
    for yi, m in zip(positions, methods):
        vals = [d[(d.config == c) & (d.model == m)]["rmse"] for c in cfgs]
        vals = [float(v.iloc[0]) for v in vals if len(v)]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [yi, yi], color="0.75", lw=1.2, zorder=1)
    base = d[(d.model == "no_mitigation") & (d.config == cfgs[0])]["rmse"]
    if len(base):
        raw_rmse = float(base.iloc[0])
        ax.axvline(raw_rmse, color="0.5", ls=":", lw=1.0, zorder=0)
        if shared_baseline:
            ax.plot(raw_rmse, present.index("no_mitigation"), "o", color="0.5", ms=6,
                    mec="white", mew=0.7, label="Shared scale-1 observation", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in present])
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    ax.margins(y=0.06)
    ax.grid(axis="y", alpha=0)


def fig_scaling(sw, out):
    d = sw[sw.experiment == "R1_scaling_method"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=True)
    for ax, sp, t in [(axes[0], "A_random_circuits", "Held-out random circuits"),
                      (axes[1], "B_depth_extrapolation", "Deeper circuits than trained on")]:
        dumbbell(ax, d[d.split == sp], ["parameter_scaling", "unitary_folding"],
                 ["Parameter scaling", "Unitary folding"],
                 ["#6f8faf", "#c2591a"], t, shared_baseline=True)
    hi = d["rmse"].max() * 1.08
    axes[0].set_xlim(0, hi)
    axes[0].legend(loc="upper right")
    for a, l in zip(axes, "ab"):
        letter(a, l, dx=-0.42 if l == "a" else -0.05)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_scaling.png"), bbox_inches="tight")
    plt.close(fig)


def fig_regime(sw, out):
    d = sw[sw.experiment == "R2_markovianity"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharex=True, sharey=True)
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
             "Transfer to the drifted calibration")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_drift.png"), bbox_inches="tight")
    plt.close(fig)


def fig_shotkernel(sw, out):
    d = sw[(sw.experiment == "R4_shot_limited_kernel") &
           (sw.model == "qk_ridge_quantum")]
    rbf = sw[(sw.experiment == "R1_scaling_method") &
             (sw.config == "parameter_scaling") & (sw.split == "A_random_circuits") &
             (sw.model == "rbf_krr")]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    fin = (d[d.shots_per_entry > 0].groupby("shots_per_entry")
           .agg(execs=("circuit_executions", "first"), rmse=("rmse", "mean"),
                lo=("rmse", "min"), hi=("rmse", "max"), n=("rmse", "size")).reset_index())
    ex = d[d.shots_per_entry == 0]
    ax.errorbar(fin["execs"], fin["rmse"], yerr=[fin["rmse"] - fin["lo"], fin["hi"] - fin["rmse"]],
                fmt="-o", color="#c2591a", mfc="white", ms=4, lw=1.4,
                capsize=2.5, elinewidth=1.0, barsabove=True,
                label=f"finite-shot fidelity ({int(fin['n'].iloc[0])} draws)")
    if len(ex):
        ax.axhline(float(ex["rmse"].iloc[0]), color="#c2591a", ls="--", lw=1.0,
                   label=f"exact Gram (simulator): {float(ex['rmse'].iloc[0]):.4f}")
    if len(rbf):
        rmse = float(rbf["rmse"].iloc[0])
        ax.axhline(rmse, color="#087e8b", ls="-.", lw=1.0,
                   label=f"classical RBF (R1 grid): {rmse:.4f}")
    base = float(d["unmitigated_rmse"].iloc[0])
    ax.axhline(base, color="0.45", ls=":", lw=1.1, label=f"no mitigation: {base:.3f}")
    ax.set_xscale("log")
    measured_entries = fin["execs"].iloc[0] / fin["shots_per_entry"].iloc[0]
    shot_axis = ax.secondary_xaxis("top", functions=(
        lambda executions: executions / measured_entries,
        lambda shots: shots * measured_entries,
    ))
    shot_axis.set_xticks(fin["shots_per_entry"],
                         [f"$10^{{{int(np.log10(s))}}}$" for s in fin["shots_per_entry"]])
    shot_axis.set_xlabel("Shots per Gram-matrix entry")
    ax.set_xlabel("Fidelity-circuit executions per draw (tuning and final fit/test)")
    ax.set_ylabel("RMSE vs ideal")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.91))
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
    summaries = [pd.read_csv(f"robust_splits_{c}.csv").groupby("model")["rmse"]
                 .agg(["mean", "std"]) for c, _, _ in cfgs]
    present = [m for m in ORDER if all(m in summary.index for summary in summaries)]
    y = np.arange(len(present))
    for k, ((_, lab, col), r) in enumerate(zip(cfgs, summaries)):
        ax.errorbar([r.loc[m, "mean"] for m in present], y + (k - 1) * 0.22,
                    xerr=[r.loc[m, "std"] for m in present], fmt="o", color=col, ms=4.5,
                    mec="white", mew=0.5, elinewidth=1.0, capsize=1.5, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in present])
    ax.set_xlabel("Test RMSE, mean $\\pm$ SD over 10 circuit-grouped splits")
    ax.set_title("Ranking over 10 splits", loc="left")
    ax.grid(axis="y", alpha=0)
    ax.legend(loc="upper right")
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
                    label="seed 7 (initialisation study)" if i == 0 else None)
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
    ticks = [0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in arms])
    ax.set_xlabel("Test RMSE on deeper circuits (Split B), log scale")
    ax.set_title("Depth extrapolation with and without structural features", loc="left")
    ax.grid(axis="y", alpha=0); ax.legend(loc="upper right", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_ood_features.png"), bbox_inches="tight")
    plt.close(fig)


def fig_nonmarkov(nm, path):
    """Idle trace distance on q0 for production and both controls, and the coupling sweep.

    Reads the record written by nonmarkov_check.py; nothing is re-simulated.
    """
    with plt.style.context("default"):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
        ax = axes[0]
        prod = nm["nonmarkovian_production"]["pairs"]
        key = max(prod, key=lambda k: prod[k]["blp"])
        x = np.arange(1, nm["n_layers"] + 1)
        for label, condition, color in (
                ("production (defect retains its state)", "nonmarkovian_production", "#c2591a"),
                ("defect reset every layer", "reset_control", "#d79a1e"),
                ("defect removed", "markovian_control", "#3f7f5f")):
            pair = nm[condition]["pairs"][key]
            ax.plot(x, pair["trace_distance"], "-o", color=color, ms=5,
                    label=f"{label}, $\\mathcal{{N}}$={pair['blp']:.2f}")
        ax.set_xlabel("Idle layer"); ax.set_ylabel("Trace distance $D(\\rho_1,\\rho_2)$")
        ax.set_xticks(x)
        ax.set_title("Trace distance revives only when the defect keeps its state",
                     loc="left", fontsize=9.5)
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1]
        sweep = nm["coupling_sweep"]
        g = [s["g_tls"] for s in sweep]; b = [s["blp_max"] for s in sweep]
        ax.plot(g, b, "-o", color="#c2591a", ms=5)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("TLS exchange coupling $g$ (rad)")
        ax.set_ylabel("Non-Markovianity $\\mathcal{N}$")
        ax.set_title("Non-Markovianity grows with coupling, zero without it",
                     loc="left", fontsize=9.5)
        ax.annotate("production\nsetting", xy=(0.15, [s["blp_max"] for s in sweep
                    if s["g_tls"] == 0.15][0]), xytext=(-6, 26),
                    textcoords="offset points", fontsize=7, ha="center", color="0.35",
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="0.5"))
        for a_, l in zip(axes, "ab"):
            a_.spines[["top", "right"]].set_visible(False)
            a_.text(-0.16, 1.06, l, transform=a_.transAxes, fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def fig_folding_seed_scaling(pairs, out):
    """Paired scaling differences for each arm, retaining the shot groups."""
    data = pairs[pairs.comparison == "folding_minus_parameter"]
    pools = sorted(data.circuit_seed.unique())
    positions = np.arange(len(ORDER))
    pool_offsets = np.linspace(-0.23, 0.23, len(pools))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.3), sharey=True)
    for ax, (split, title) in zip(axes, SEED_SPLITS):
        group = data[data.split == split]
        for pool, color, offset in zip(pools, cycle(POOL_COLORS), pool_offsets):
            rows = group[group.circuit_seed == pool]
            for position, model in zip(positions, ORDER):
                draws = rows.loc[rows.model == model, "diff"].to_numpy()
                height = position + offset
                ax.hlines(height, draws.min(), draws.max(), color=color, lw=1.1)
                ax.plot(draws, np.full(draws.size, height), "o", color=color,
                        ms=3.6, alpha=0.82, label=str(pool) if position == 0 else None)
        ax.axvline(0, color="0.3", lw=0.8, ls="--")
        ax.set_yticks(positions, [NAME[model] for model in ORDER])
        ax.tick_params(labelsize=8.3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Folded RMSE $-$ parameter-scaled RMSE")
        ax.grid(axis="y", alpha=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Circuit seed", loc="upper center", ncol=len(pools))
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(os.path.join(out, "fig_ch5_folding_seed_scaling.png"), bbox_inches="tight")
    plt.close(fig)


def fig_folding_seed_kernels(pairs, out):
    """Display each seed cell's circuit-bootstrap interval for RBF minus fidelity."""
    data = pairs[pairs.comparison.str.endswith(":rbf_minus_fidelity")]
    pools = sorted(data.circuit_seed.unique())
    shot_seeds = sorted(data.shot_seed.unique())
    pool_positions = np.arange(len(pools))
    shot_offsets = np.linspace(-0.25, 0.25, len(shot_seeds))
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.0), sharex="row", sharey=True)
    for row, (split, split_title) in enumerate(SEED_SPLITS):
        for col, (protocol, protocol_title) in enumerate(SEED_PROTOCOLS):
            ax = axes[row, col]
            group = data[(data.split == split) & (data.comparison == f"{protocol}:rbf_minus_fidelity")]
            for pool, color, height in zip(pools, cycle(POOL_COLORS), pool_positions):
                draws = group[group.circuit_seed == pool].set_index("shot_seed")
                for shot, marker, offset in zip(shot_seeds, cycle(SHOT_MARKERS), shot_offsets):
                    point = draws.loc[shot]
                    y = height + offset
                    ax.hlines(y, point.diff_lo, point.diff_hi, color=color, lw=1.1)
                    ax.plot(point["diff"], y, marker, color=color, ms=4.2,
                            mfc=color if point.resolved else "white", zorder=3)
            ax.axvline(0, color="0.3", lw=0.8, ls="--")
            ax.set_yticks(pool_positions, [str(pool) for pool in pools])
            ax.tick_params(labelsize=8.3)
            ax.set_ylim(len(pools) - 0.55, -0.45)
            ax.set_title(f"{split_title} · {protocol_title}", loc="left", fontsize=8.5)
            ax.grid(axis="y", alpha=0)
        axes[row, 0].set_ylabel("Circuit seed")
    for shot, marker in zip(shot_seeds, cycle(SHOT_MARKERS)):
        axes[0, 0].plot([], [], marker, color="0.3", label=str(shot), ms=4.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Shot seed", loc="upper center", ncol=len(shot_seeds))
    fig.supxlabel("Classical RBF RMSE $-$ quantum fidelity RMSE", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.01, 1, 0.89))
    fig.savefig(os.path.join(out, "fig_ch5_folding_seed_kernels.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--folding-seed-pairs", help="plot the circuit-pool and shot-draw replication")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    style()
    if a.folding_seed_pairs:
        pairs = pd.read_csv(a.folding_seed_pairs)
        fig_folding_seed_scaling(pairs, a.outdir)
        fig_folding_seed_kernels(pairs, a.outdir)
    else:
        sw = pd.read_csv("sweep_realistic.csv")
        fig_scaling(sw, a.outdir)
        fig_regime(sw, a.outdir)
        fig_drift(sw, a.outdir)
        fig_shotkernel(sw, a.outdir)
        fig_robustness(a.outdir)
        fig_ood_features(a.outdir)
        with open("nonmarkovianity_check.json") as source:
            fig_nonmarkov(json.load(source), os.path.join(a.outdir, "fig_ch5_nonmarkov.png"))
    print("wrote:", sorted(f for f in os.listdir(a.outdir)
                           if f.startswith("fig_ch5_") and f.endswith(".png")))


if __name__ == "__main__":
    main()
