"""Figures for the device-realistic regime (Chapter 6).

Reads sweep_realistic.csv and writes:
  fig_ch5_scaling.png   parameter scaling vs unitary folding, both splits
  fig_ch5_regime.png    memory: production vs reset (memoryless) and defect-free controls
  fig_ch5_robustness.png ranking over 10 grouped splits; VQC seed spread
  fig_ch5_ood_features.png depth extrapolation with and without structural features
  fig_ch5_drift.png     calibration drift
  fig_ch5_shotkernel.png shot-limited quantum kernel: error vs tuning and final execution costs

Use --folding-seed-pairs for the initial circuit/shot replication and
--folding-depth-pairs with --folding-depth-summary for the depth expansion.
Use --folding-depth-retuned with --folding-depth-previous to compare the two
kernel tuning protocols on matched depth holdouts.
Use --folding-depth-validation to plot the fresh locked cohort alongside
the earlier depth-aware estimate.
Use --folding-depth-shot to plot the locked folded depth holdout across
fidelity-kernel shot budgets.
Use --folding-depth-shot-crossover to compare fixed and retuned models on
the same folded depth holdouts across new kernel-shot seeds.
Use --folding-depth-shot-fixed to plot the fixed-parameter sweep on fresh pools.
Use --folding-depth-shot-fixed-rbf to compare the matched RBF baseline.
Use --folding-depth-shot-fixed-both to compare the kernels with fixed settings.
Use --folding-depth-shot-recalibrated for matched earlier-pool selection.
Use --folding-depth-shot-budget-calibrated for selection by finite shot budget.

Usage: python make_figures_realistic.py --outdir <dir>
"""
import argparse
from itertools import cycle
import os

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in arms])
    ax.set_xlabel("Test RMSE on deeper circuits (Split B), log scale")
    ax.set_title("Depth extrapolation with and without structural features", loc="left")
    ax.grid(axis="y", alpha=0); ax.legend(loc="upper right", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_ood_features.png"), bbox_inches="tight")
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


def fig_folding_depth_pools(pairs, summary, out, *,
                           cohorts=(("all_pools", "All pools"), ("new_pools", "New pools")),
                           filename="fig_ch5_folding_depth_pools.png"):
    """Paired kernel effects by independent pool and cohort."""
    effects = pairs[pairs.comparison.str.endswith(":rbf_minus_fidelity")]
    pools = sorted(effects.circuit_seed.unique())
    positions = np.arange(len(pools))
    cohort_positions = [len(pools) + 0.15, len(pools) + 1.15]
    colors = ["#0072b2", "#d55e00"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 5.7), sharey=True)
    for ax, (protocol, title), color in zip(axes, SEED_PROTOCOLS, colors):
        comparison = f"{protocol}:rbf_minus_fidelity"
        draws = effects[effects.comparison == comparison]
        estimates = summary[summary.comparison == comparison].set_index("cohort")
        for y, pool in zip(positions, pools):
            shot_differences = draws.loc[draws.circuit_seed == pool, "diff"].to_numpy()
            ax.hlines(y, shot_differences.min(), shot_differences.max(), lw=0.9, color="0.62")
            ax.plot(shot_differences, np.full(shot_differences.size, y), "o",
                    ms=3.0, color="0.53", alpha=0.72, zorder=3)
            ax.plot(shot_differences.mean(), y, "o", color=color, ms=5.0, zorder=4)
        for y, (cohort, _) in zip(cohort_positions, cohorts):
            r = estimates.loc[cohort]
            ax.hlines(y, r.ci95_t_lo, r.ci95_t_hi, lw=1.6, color=color)
            ax.plot(r.diff_mean, y, "D", color=color, ms=5.2, zorder=4,
                    mfc=color if cohort == "all_pools" else "white")
        ax.axhline(len(pools) - 0.46, lw=0.65, color="0.78")
        ax.axvline(0, color="0.32", lw=0.9, ls="--")
        cohort_labels = [f"{label} ({int(estimates.loc[cohort, 'n_pools'])})"
                         for cohort, label in cohorts]
        ax.set_yticks([*positions, *cohort_positions], [*map(str, pools), *cohort_labels])
        ax.set_ylim(cohort_positions[-1] + 0.65, -0.6)
        ax.set_title(title, loc="left")
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", alpha=0)
    axes[0].set_ylabel("Circuit seed or cohort")
    handles = [
        Line2D([], [], marker="o", ms=3, lw=0.9, color="0.53", label="Shot draws"),
        Line2D([], [], marker="o", ms=5, lw=0, color="0.25", label="Pool mean"),
        Line2D([], [], marker="D", ms=5, lw=1.6, color="0.25", label="Cohort mean and 95% interval"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles))
    fig.supxlabel("Classical RBF RMSE $-$ quantum fidelity RMSE", fontsize=9)
    fig.tight_layout(rect=(0, 0.015, 1, 0.94))
    fig.savefig(os.path.join(out, filename), bbox_inches="tight")
    plt.close(fig)


def fig_folding_depth_retuning(previous_pools, depth_aware_pools, previous_summary,
                               depth_aware_summary, out):
    """Compare RBF minus fidelity on paired pools before and after retuning."""
    labels = ["Mixed-depth tuning", "Depth-aware tuning"]
    colors = ["#0072b2", "#d55e00"]
    markers = ["o", "s"]
    offsets = [-0.16, 0.16]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 5.45), sharey=True)
    for ax, (protocol, title) in zip(axes, SEED_PROTOCOLS):
        comparison = f"{protocol}:rbf_minus_fidelity"
        pool_values = [frame.loc[frame.comparison == comparison].set_index(
            "circuit_seed")["diff_mean"].sort_index()
            for frame in (previous_pools, depth_aware_pools)]
        if not pool_values[0].index.is_unique or not pool_values[0].index.equals(pool_values[1].index):
            raise ValueError(f"Unmatched circuit pools for {protocol}")
        pools = pool_values[0].index
        positions = np.arange(len(pools))
        cohorts = ["all_pools", "new_pools"]
        cohort_positions = np.array([len(pools) + 0.1, len(pools) + 1.1])
        for y, before, after in zip(positions, *pool_values):
            ax.plot([before, after], y + np.array(offsets), color="0.76", lw=0.8, zorder=1)
        for values, summary, color, marker, offset in zip(
                pool_values, (previous_summary, depth_aware_summary), colors, markers, offsets):
            ax.plot(values, positions + offset, marker, color=color, ms=4, zorder=3)
            estimates = summary.loc[summary.comparison == comparison].set_index("cohort").loc[cohorts]
            y = cohort_positions + offset
            ax.hlines(y, estimates.ci95_t_lo, estimates.ci95_t_hi, color=color, lw=1.5)
            ax.plot(estimates.diff_mean, y, marker, color=color, ms=4.5, zorder=3)
        ax.axhline(len(pools) - 0.47, lw=0.6, color="0.78")
        ax.axvline(0, color="0.32", lw=0.9, ls="--")
        cohort_labels = [f"{label} ({int(estimates.loc[cohort, 'n_pools'])})"
                         for label, cohort in zip(["All pools", "Additional pools"], cohorts)]
        ax.set_yticks([*positions, *cohort_positions],
                      [*map(str, pools), *cohort_labels])
        ax.set_ylim(cohort_positions[-1] + 0.68, -0.62)
        ax.set_title(title, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", alpha=0)
    axes[0].set_ylabel("Circuit seed or cohort")
    handles = [Line2D([], [], marker=marker, ms=4.5, ls="", color=color, label=label)
               for marker, color, label in zip(markers, colors, labels)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles))
    fig.supxlabel("Classical RBF RMSE $-$ quantum fidelity RMSE", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(os.path.join(out, "fig_ch5_folding_depth_retuning.png"), bbox_inches="tight")
    plt.close(fig)


def fig_folding_depth_influential(previous_scores, depth_aware_scores, out):
    """Show the matched shot draws on the influential folded pool."""
    seed = 2026112525
    draw_colors = ["#0072b2", "#d55e00", "#00845f"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.7), sharey=True)
    for ax, (model, title) in zip(axes, [("rbf_krr", "Classical RBF"),
                                        ("qk_ridge_quantum", "Quantum fidelity")]):
        rmse = [frame.loc[(frame.config == "unitary_folding") & (frame.model == model) &
                          (frame.circuit_seed == seed)].set_index(
                              "shot_seed")["rmse"].sort_index()
                for frame in (previous_scores, depth_aware_scores)]
        if (len(rmse[0]) != 3 or not rmse[0].index.is_unique
                or not rmse[0].index.equals(rmse[1].index)):
            raise ValueError(f"Unmatched shot draws for pool {seed} and model {model}")
        for i, (color, marker, before, after) in enumerate(zip(draw_colors, SHOT_MARKERS, *rmse), 1):
            ax.plot([0, 1], [before, after], color=color, marker=marker, ms=4.8,
                    lw=0.95, label=f"Draw {i}")
        ax.set_xticks([0, 1], ["Mixed-depth tuning", "Depth-aware tuning"])
        ax.set_xlim(-0.15, 1.15)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0)
    axes[0].set_ylabel("Depth-holdout RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(out, "fig_ch5_folding_depth_influential.png"), bbox_inches="tight")
    plt.close(fig)


def fig_folding_depth_shot(scores, pool_summary, summary, out):
    """Compare shot-estimated fidelity with its matched folded depth references."""
    budgets = [100, 1000, 10000, 100000, 0]
    labels = [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", "Exact"]
    positions = np.arange(len(budgets))
    fidelity = scores[scores.model == "qk_ridge_quantum"]
    pool_error = fidelity.groupby(["shots_per_entry", "circuit_seed"])["rmse"].mean()
    error = pool_error.groupby("shots_per_entry").mean().loc[budgets]
    contrast_pools = pool_summary[pool_summary.comparison == "rbf_minus_fidelity"]
    contrast = summary[summary.comparison == "rbf_minus_fidelity"].set_index(
        "shots_per_entry").loc[budgets]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.9, 3.75))
    color = "#a33f12"
    for x, budget in zip(positions, budgets):
        values = pool_error.loc[budget].to_numpy()
        effects = contrast_pools.loc[contrast_pools.shots_per_entry == budget, "diff_mean"]
        for ax, points in ((left, values), (right, effects)):
            ax.scatter(np.full(len(points), x), points, s=10, color="0.53", alpha=0.46, zorder=2)
    left.plot(positions[:-1], error.iloc[:-1], "-o", color=color, ms=4.6, lw=1.25, zorder=3,
              label="Fidelity across pools")
    left.plot(positions[-1], error.iloc[-1], "D", color=color, ms=4.6, zorder=3)
    for model, label, style in (
            ("rbf_krr", "Classical RBF", "--"),
            ("zne_linear", "Linear ZNE", "-."),
            ("no_mitigation", "No mitigation", ":")):
        reference = scores.loc[scores.model == model].groupby("circuit_seed")["rmse"].mean().mean()
        left.axhline(reference, color="#09737b" if model == "rbf_krr" else "0.37",
                     ls=style, lw=1.05, label=label)
    right.axhline(0, color="0.37", ls="--", lw=1)
    right.vlines(positions, contrast.ci95_t_lo, contrast.ci95_t_hi, color=color, lw=1.45, zorder=3)
    right.plot(positions[:-1], contrast.diff_mean.iloc[:-1], "o", color=color, ms=4.6, zorder=4)
    right.plot(positions[-1], contrast.diff_mean.iloc[-1], "D", color=color, ms=4.6, zorder=4)
    left.set_title("a   Folded depth-holdout error", loc="left")
    right.set_title("b   Paired kernel comparison", loc="left")
    left.set_ylabel("RMSE vs ideal")
    right.set_ylabel("RBF RMSE − fidelity RMSE")
    for ax in (left, right):
        ax.axvline(3.5, color="0.70", lw=0.8)
        ax.set_xticks(positions, labels)
        ax.set_xlim(-0.38, positions[-1] + 0.38)
        ax.set_xlabel("Fidelity shots per entry")
        ax.grid(axis="x", alpha=0)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(os.path.join(out, "fig_ch5_folding_depth_shot.png"), bbox_inches="tight")
    plt.close(fig)


def fig_folding_depth_shot_crossover(pool_summary, summary, out):
    """Show paired budget and penalty effects with circuit-pool uncertainty."""
    estimates = summary.loc[summary.cohort == "all_kernel_seeds"].set_index("comparison")
    pools = pool_summary.loc[pool_summary.cohort == "all_kernel_seeds"]
    panels = [
        ("a   Higher shot budget", "RMSE at $10^5$ minus RMSE at $10^4$", [
            ("budget_gap_retuned", "Retuned"),
            ("budget_gap_fixed_alpha_0.1", r"Fixed $\alpha=0.1$"),
            ("budget_gap_fixed_alpha_0.01", r"Fixed $\alpha=0.01$"),
        ]),
        ("b   Penalty changes and residual", "Paired RMSE difference", [
            ("alpha_gap_10000", r"$\alpha=0.01$ minus $0.1$ at $10^4$"),
            ("alpha_gap_100000", r"$\alpha=0.01$ minus $0.1$ at $10^5$"),
            ("retuned_gap_minus_fixed_switch_gap", "Retuned gap minus\nfixed switch gap"),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.85), layout="constrained")
    color = "#a33f12"
    for ax, (title, xlabel, comparisons) in zip(axes, panels):
        positions = np.arange(len(comparisons))
        names, labels = zip(*comparisons)
        aggregate = estimates.loc[list(names)]
        for y, name in zip(positions, names):
            values = pools.loc[pools.comparison == name, "diff_mean"]
            ax.scatter(values, np.full(len(values), y + 0.19), s=11, color="0.47", alpha=0.58, zorder=2)
        errors = np.stack([
            aggregate.diff_mean - aggregate.ci95_t_lo,
            aggregate.ci95_t_hi - aggregate.diff_mean,
        ])
        ax.errorbar(aggregate.diff_mean, positions, xerr=errors, fmt="|", color=color,
                    markersize=7, markeredgewidth=0.9, elinewidth=1.25, capsize=2, zorder=3)
        ax.axvline(0, color="0.37", lw=0.9, ls="--")
        ax.set_yticks(positions, labels)
        ax.set_ylim(positions[-1] + 0.52, -0.38)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", alpha=0)
    fig.savefig(os.path.join(out, "fig_ch5_folding_depth_shot_crossover.png"), bbox_inches="tight")
    plt.close(fig)


def fig_folding_depth_shot_fixed(pool_summary, summary, out):
    """Plot absolute fidelity RMSE and paired budget changes on fresh pools."""
    aggregate = summary.loc[summary.cohort == "all_kernel_seeds"].set_index("comparison")
    pools = pool_summary.loc[pool_summary.cohort == "all_kernel_seeds"]
    budgets = [100, 1000, 10000, 100000]
    budget_labels = [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]
    absolute_names = [f"absolute_rmse_{budget}" for budget in budgets] + ["absolute_rmse_exact"]
    adjacent_names = [f"budget_gap_{hi}_minus_{lo}" for lo, hi in zip(budgets, budgets[1:])]
    adjacent_labels = [f"{hi} − {lo}" for lo, hi in zip(budget_labels, budget_labels[1:])]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.9, 3.45), layout="constrained")
    color = "#a33f12"
    for ax, comparisons, labels in (
        (left, absolute_names, [*budget_labels, "Exact"]),
        (right, adjacent_names, adjacent_labels),
    ):
        positions = np.arange(len(comparisons))
        estimates = aggregate.loc[comparisons]
        for x, name in zip(positions, comparisons):
            values = pools.loc[pools.comparison == name, "diff_mean"]
            ax.scatter(np.full(len(values), x), values, s=13, color="0.47", alpha=0.56, zorder=2)
        errors = np.stack([
            estimates.diff_mean - estimates.ci95_t_lo,
            estimates.ci95_t_hi - estimates.diff_mean,
        ])
        ax.errorbar(positions, estimates.diff_mean, yerr=errors, fmt="_", markersize=7,
                    markeredgewidth=0.9, elinewidth=1.25, capsize=2, color=color, zorder=4)
        ax.set_xticks(positions, labels)
        ax.set_xlim(-0.33, positions[-1] + 0.33)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x", alpha=0)

    left.plot(np.arange(len(budgets)), aggregate.loc[absolute_names[:-1], "diff_mean"],
              color=color, lw=1.15, zorder=3)
    left.axvline(len(budgets) - 0.5, color="0.66", lw=0.8)
    left.set_title("a   Fixed-parameter fidelity error", loc="left")
    left.set(xlabel="Fidelity shots per entry", ylabel="Depth-holdout RMSE")
    right.axhline(0, color="0.37", lw=0.9, ls="--")
    right.set_title("b   Paired change at each budget step", loc="left")
    right.set(xlabel="Higher minus lower shots per entry", ylabel="Paired RMSE difference")
    fig.savefig(os.path.join(out, "fig_ch5_folding_depth_shot_fixed.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_rbf_fidelity_differences(ax, pool_summary, summary, budgets, budget_labels, color):
    comparisons = [f"rbf_minus_fixed_fidelity_{b}" for b in [*budgets, "exact"]]
    aggregate = summary.loc[summary.cohort == "all_kernel_seeds"].set_index("comparison")
    effects = aggregate.loc[comparisons]
    pools = pool_summary.loc[pool_summary.cohort == "all_kernel_seeds"]
    all_positions = np.arange(len(comparisons))
    for x, comparison in zip(all_positions, comparisons):
        values = pools.loc[pools.comparison == comparison, "diff_mean"]
        ax.scatter(np.full(len(values), x), values, s=13, color="0.47", alpha=0.56, zorder=2)
    errors = np.stack([
        effects.diff_mean - effects.ci95_t_lo, effects.ci95_t_hi - effects.diff_mean,
    ])
    ax.errorbar(all_positions, effects.diff_mean, yerr=errors, fmt="_", markersize=7,
                markeredgewidth=0.9, elinewidth=1.25, capsize=2, color=color, zorder=3)
    ax.axhline(0, color="0.37", lw=0.9, ls="--")
    ax.axvline(len(budgets) - 0.5, color="0.66", lw=0.8)
    ax.set(ylabel="RBF RMSE minus fidelity RMSE", xlabel="Fidelity shots per entry")
    ax.set_xticks(all_positions, [*budget_labels, "Exact"])
    ax.set_xlim(-0.33, all_positions[-1] + 0.33)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def fig_folding_depth_shot_fixed_rbf(pool_summary, summary, costs, out):
    """Show paired RBF RMSE and measured executions for a new prediction."""
    budgets = [100, 1000, 10000, 100000]
    budget_labels = [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]
    positions = np.arange(len(budgets))
    fidelity_color, rbf_color = "#a33f12", "#006b72"

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.9, 3.2), layout="constrained")
    _plot_rbf_fidelity_differences(left, pool_summary, summary, budgets, budget_labels, fidelity_color)
    left.set_title("a   Paired depth-holdout error", loc="left")

    cost = costs.set_index(["model", "shots_per_entry"])["measured_executions_new_row_mean"]
    rbf = cost.loc["rbf_krr", 0]
    fidelity = cost.loc["qk_ridge_quantum"].loc[budgets]
    right.plot(positions, fidelity, "o-", color=fidelity_color, lw=1.15, ms=4, label="Sampled fidelity")
    right.axhline(rbf, color=rbf_color, ls="--", lw=1.15, label="Classical RBF")
    right.set_title("b   Circuit executions per prediction", loc="left")
    right.set(ylabel="Observed-feature and kernel executions", xlabel="Fidelity shots per entry", yscale="log")
    right.set_xticks(positions, budget_labels)
    right.set_xlim(-0.33, positions[-1] + 0.33)
    right.legend(loc="upper left")
    for ax in (left, right):
        ax.grid(axis="x", alpha=0)
    fig.savefig(os.path.join(out, "fig_ch5_folding_depth_shot_fixed_rbf.png"), bbox_inches="tight")
    plt.close(fig)


def fig_folding_depth_shot_fixed_both(pool_summary, summary, fixed, out, *,
                                     filename="fig_ch5_folding_depth_shot_fixed_both.png",
                                     zoom_exact=False, fidelity_label="Fixed fidelity"):
    """Compare fixed kernel models on the same fresh depth holdouts."""
    budgets = [100, 1000, 10000, 100000]
    budget_labels = [r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"]
    names = [f"absolute_rmse_{budget}" for budget in [*budgets, "exact"]]
    positions = np.arange(len(names))
    fidelity_color, rbf_color = "#a33f12", "#006b72"

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.9, 3.2), layout="constrained")
    fidelity = fixed.loc[fixed.cohort == "all_kernel_seeds"].set_index("comparison").loc[names]
    rbf = summary.loc[(summary.cohort == "all_kernel_seeds") &
                      (summary.comparison == "absolute_rmse_rbf")].iloc[0]
    errors = np.stack([
        fidelity.diff_mean - fidelity.ci95_t_lo, fidelity.ci95_t_hi - fidelity.diff_mean,
    ])
    left.axhspan(rbf.ci95_t_lo, rbf.ci95_t_hi, color=rbf_color, alpha=0.13, lw=0, zorder=1)
    left.axhline(rbf.diff_mean, color=rbf_color, ls="--", lw=1.15, label="Locked RBF", zorder=2)
    left.plot(positions[:-1], fidelity.diff_mean.iloc[:-1], color=fidelity_color,
              lw=1.15, label=fidelity_label, zorder=3)
    left.errorbar(positions, fidelity.diff_mean, yerr=errors, fmt="_", markersize=7,
                  markeredgewidth=0.9, elinewidth=1.25, capsize=2, color=fidelity_color, zorder=4)
    left.axvline(len(budgets) - 0.5, color="0.66", lw=0.8)
    left.set_title("a   Mean depth-holdout RMSE", loc="left")
    left.set(xlabel="Fidelity shots per entry", ylabel="Held-out RMSE")
    left.set_xticks(positions, [*budget_labels, "Exact"])
    left.set_xlim(-0.33, positions[-1] + 0.33)
    left.yaxis.set_major_locator(MaxNLocator(nbins=5))
    left.legend(loc="upper right")

    _plot_rbf_fidelity_differences(right, pool_summary, summary, budgets, budget_labels, fidelity_color)
    right.set_title("b   Paired depth-holdout error", loc="left")
    if zoom_exact:
        contrast = "rbf_minus_fixed_fidelity_exact"
        exact = summary.loc[(summary.cohort == "all_kernel_seeds") & (summary.comparison == contrast)].iloc[0]
        pools = pool_summary.loc[(pool_summary.cohort == "all_kernel_seeds") &
                                 (pool_summary.comparison == contrast), "diff_mean"]
        inset = right.inset_axes([0.50, 0.10, 0.43, 0.39])
        inset.scatter(np.zeros(len(pools)), pools, s=10, color="0.47", alpha=0.56, zorder=2)
        inset.errorbar(0, exact.diff_mean, fmt="_", markersize=7, elinewidth=1.25,
                       yerr=[[exact.diff_mean - exact.ci95_t_lo], [exact.ci95_t_hi - exact.diff_mean]],
                       capsize=2, color=fidelity_color, zorder=3)
        inset.axhline(0, color="0.37", lw=0.8, ls="--")
        inset.set_title("Exact entry", loc="left", fontsize=8)
        inset.set(xticks=[], xlim=(-0.45, 0.45))
        inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
        inset.tick_params(axis="y", labelsize=7)
    for ax in (left, right):
        ax.grid(axis="x", alpha=0)
    fig.savefig(os.path.join(out, filename), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--folding-seed-pairs", help="plot the initial circuit/shot seed replication")
    modes.add_argument("--folding-depth-pairs", help="plot the depth expansion")
    modes.add_argument("--folding-depth-retuned", help="directory containing depth-aware results")
    modes.add_argument("--folding-depth-validation", help="directory containing fresh locked validation results")
    modes.add_argument("--folding-depth-shot", help="directory containing the folded-depth shot-budget results")
    modes.add_argument("--folding-depth-shot-crossover", help="directory containing the folded-depth kernel-shot crossover results")
    modes.add_argument("--folding-depth-shot-fixed", help="directory containing the fixed-parameter folded-depth sweep results")
    modes.add_argument("--folding-depth-shot-fixed-rbf", help="directory containing matched RBF contrasts and circuit costs")
    modes.add_argument("--folding-depth-shot-fixed-both", help="directory containing the locked-kernel accuracy contrasts")
    modes.add_argument("--folding-depth-shot-recalibrated", help="directory containing the matched earlier-pool selection results")
    modes.add_argument("--folding-depth-shot-budget-calibrated", help="directory containing the results for selection by budget")
    ap.add_argument("--folding-depth-summary", help="pool-level intervals for the depth expansion")
    ap.add_argument("--folding-depth-previous", help="directory containing mixed-depth tuning results")
    ap.add_argument("--folding-depth-context", default="results",
                    help="directory containing the corresponding archived folded-depth results")
    a = ap.parse_args()
    if bool(a.folding_depth_pairs) != bool(a.folding_depth_summary):
        ap.error("--folding-depth-pairs and --folding-depth-summary must be supplied together")
    if bool(a.folding_depth_retuned) != bool(a.folding_depth_previous):
        ap.error("--folding-depth-retuned and --folding-depth-previous must be supplied together")
    os.makedirs(a.outdir, exist_ok=True)
    style()
    if a.folding_seed_pairs:
        pairs = pd.read_csv(a.folding_seed_pairs)
        fig_folding_seed_scaling(pairs, a.outdir)
        fig_folding_seed_kernels(pairs, a.outdir)
    elif a.folding_depth_pairs:
        fig_folding_depth_pools(
            pd.read_csv(a.folding_depth_pairs), pd.read_csv(a.folding_depth_summary), a.outdir,
        )
    elif a.folding_depth_retuned:
        previous = os.path.join(a.folding_depth_previous, "folding_depth")
        retuned = os.path.join(a.folding_depth_retuned, "folding_depth_aware")
        fig_folding_depth_retuning(
            pd.read_csv(previous + "_pool_summary.csv"), pd.read_csv(retuned + "_pool_summary.csv"),
            pd.read_csv(previous + "_summary.csv"), pd.read_csv(retuned + "_summary.csv"), a.outdir,
        )
        fig_folding_depth_influential(
            pd.read_csv(previous + "_scores.csv"), pd.read_csv(retuned + "_scores.csv"), a.outdir,
        )
    elif a.folding_depth_validation:
        fresh = os.path.join(a.folding_depth_validation, "folding_depth_validation")
        earlier = pd.read_csv(os.path.join(a.folding_depth_context, "folding_depth_aware_summary.csv"))
        earlier = earlier.loc[earlier.cohort == "all_pools"].assign(cohort="earlier_pools")
        summary = pd.concat([pd.read_csv(fresh + "_summary.csv"), earlier], ignore_index=True)
        fig_folding_depth_pools(
            pd.read_csv(fresh + "_pairs.csv"), summary, a.outdir,
            cohorts=(("all_pools", "Fresh validation"), ("earlier_pools", "Earlier tuning study")),
            filename="fig_ch5_folding_depth_validation.png",
        )
    elif a.folding_depth_shot:
        stem = os.path.join(a.folding_depth_shot, "folding_depth_shot")
        fig_folding_depth_shot(
            pd.read_csv(stem + "_scores.csv"), pd.read_csv(stem + "_pool_summary.csv"),
            pd.read_csv(stem + "_summary.csv"), a.outdir,
        )
    elif a.folding_depth_shot_crossover:
        stem = os.path.join(a.folding_depth_shot_crossover, "folding_depth_shot_crossover")
        fig_folding_depth_shot_crossover(
            pd.read_csv(stem + "_pool_summary.csv"), pd.read_csv(stem + "_summary.csv"), a.outdir,
        )
    elif a.folding_depth_shot_fixed:
        stem = os.path.join(a.folding_depth_shot_fixed, "folding_depth_shot_fixed")
        fig_folding_depth_shot_fixed(
            pd.read_csv(stem + "_pool_summary.csv"), pd.read_csv(stem + "_summary.csv"), a.outdir,
        )
    elif a.folding_depth_shot_fixed_rbf:
        stem = os.path.join(a.folding_depth_shot_fixed_rbf, "folding_depth_shot_fixed_rbf")
        fig_folding_depth_shot_fixed_rbf(
            pd.read_csv(stem + "_pool_summary.csv"), pd.read_csv(stem + "_summary.csv"),
            pd.read_csv(stem + "_cost_summary.csv"), a.outdir,
        )
    elif a.folding_depth_shot_fixed_both:
        stem = os.path.join(a.folding_depth_shot_fixed_both, "folding_depth_shot_fixed_both")
        fig_folding_depth_shot_fixed_both(
            pd.read_csv(stem + "_pool_summary.csv"), pd.read_csv(stem + "_summary.csv"),
            pd.read_csv(os.path.join(a.folding_depth_context, "folding_depth_shot_fixed_summary.csv")),
            a.outdir,
        )
    elif a.folding_depth_shot_recalibrated:
        stem = os.path.join(a.folding_depth_shot_recalibrated, "folding_depth_shot_recalibrated")
        summary = pd.read_csv(stem + "_summary.csv")
        fig_folding_depth_shot_fixed_both(
            pd.read_csv(stem + "_pool_summary.csv"), summary, summary, a.outdir,
            filename="fig_ch5_folding_depth_shot_recalibrated.png", zoom_exact=True,
        )
    elif a.folding_depth_shot_budget_calibrated:
        stem = os.path.join(a.folding_depth_shot_budget_calibrated, "folding_depth_shot_budget_calibrated")
        summary = pd.read_csv(stem + "_summary.csv")
        fig_folding_depth_shot_fixed_both(
            pd.read_csv(stem + "_pool_summary.csv"), summary, summary, a.outdir,
            filename="fig_ch5_folding_depth_shot_budget_calibrated.png",
            fidelity_label="Budget-calibrated fidelity",
        )
    else:
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
