"""Generate every figure in Chapter 5 of the thesis.

Usage: python make_figures.py --outdir <dir>
Reads benchmark_results.csv, breakdown_splitA.csv, sweep_results.csv and
bandwidth_study.csv; writes PNG at 300 dpi.
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAME = {"no_mitigation": "No mitigation", "zne_linear": "ZNE, linear",
        "zne_richardson2": "ZNE, Richardson (deg 2)", "zne_exponential": "ZNE, exponential",
        "ridge": "Ridge regression", "hist_gbm": "Gradient-boosted trees",
        "mlp": "Neural net (64,64)", "rbf_krr": "Classical RBF kernel ridge",
        "qk_ridge_quantum": "Quantum kernel ridge",
        "dr_vqc_quantum": "Data-reuploading VQC",
        "bounded_vqc_quantum": "Bounded-encoding VQC"}
KIND_C = {"unmitigated": "0.45", "analytic ZNE": "#6f8faf",
          "classical ML": "#3f7f5f", "hybrid quantum-classical": "#c2591a"}
# per-model colours for line plots: hue family follows the kind, samples within it
MODEL_C = {"no_mitigation": "0.45", "zne_linear": "#9db4cc",
           "zne_richardson2": "#6f8faf", "zne_exponential": "#4a6a8a",
           "ridge": "#7fae97", "hist_gbm": "#3f7f5f", "mlp": "#1f5136",
           "rbf_krr": "#087e8b",
           "qk_ridge_quantum": "#c2591a", "dr_vqc_quantum": "#8c4a2f",
           "bounded_vqc_quantum": "#e08b4e"}


def style(sizes=(9, 8, 7)):
    base, ann, tick = sizes
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": ann, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.grid": False, "legend.frameon": False, "lines.linewidth": 1.4,
        "lines.markersize": 4.5, "figure.constrained_layout.use": False,
    })


def letter(ax, ch, dx=-0.16, dy=1.03):
    ax.text(dx, dy, ch, transform=ax.transAxes, fontweight="bold",
            fontsize=mpl.rcParams["axes.titlesize"] + 1, va="bottom", ha="left")


# ------------------------------------------------------------------ figure 1
def fig_main(res, out):
    order = (res[res.split == "A_random_circuits"]
             .sort_values("rmse", ascending=False)["model"].tolist())
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), sharey=True)
    for ax, (sp, title) in zip(axes, [("A_random_circuits", "Held-out random circuits"),
                                      ("B_depth_extrapolation",
                                       "Deeper circuits than trained on")]):
        d = res[res.split == sp].set_index("model").loc[order]
        y = np.arange(len(order))
        ax.axvline(d.loc["no_mitigation", "rmse"], color="0.45", ls=":", lw=1.1, zorder=1)
        for yi, (m, r) in zip(y, d.iterrows()):
            c = KIND_C[r["kind"]]
            ax.hlines(yi, 0, r["rmse"], color=c, lw=1.0, alpha=0.45, zorder=2)
            ax.plot([r["rmse_lo"], r["rmse_hi"]], [yi, yi], color=c, lw=2.6,
                    alpha=0.45, solid_capstyle="butt", zorder=3)
            ax.plot(r["rmse"], yi, "o", color=c, ms=6.5, mec="white", mew=0.8, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in order])
        ax.set_title(title)
        ax.set_xlabel("RMSE vs ideal expectation value  (lower = better)")
        ax.set_xlim(0, max(0.21, d["rmse_hi"].max() * 1.12))
        ax.margins(y=0.06)
    axes[1].annotate("VQC fails to extrapolate",
                     xy=(res[(res.split == "B_depth_extrapolation") &
                             (res.model == "dr_vqc_quantum")]["rmse"].iloc[0],
                         order.index("dr_vqc_quantum")),
                     xytext=(-34, 30), textcoords="offset points", fontsize=7,
                     color=KIND_C["hybrid quantum-classical"], ha="center",
                     arrowprops=dict(arrowstyle="-", lw=0.8,
                                     color=KIND_C["hybrid quantum-classical"]))
    h = [mpl.lines.Line2D([], [], color=c, marker="o", ls="-", ms=5.5, label=k)
         for k, c in KIND_C.items()]
    h.append(mpl.lines.Line2D([], [], color="0.45", ls=":", lw=1.1,
                              label="unmitigated error level"))
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=7, handlelength=1.4,
               columnspacing=1.4, bbox_to_anchor=(0.5, -0.055))
    for a, l in zip(axes, "ab"):
        letter(a, l, dx=-0.55)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "fig_ch5_main.png"))
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_breakdown(brk, out):
    sel = ["no_mitigation", "zne_richardson2", "hist_gbm", "rbf_krr", "qk_ridge_quantum",
           "dr_vqc_quantum"]
    sel = [m for m in sel if m in set(brk.model)]
    obs_order = ["Z0", "Z0Z1", "X0X1", "Z1Z2Z3", "Z_all"]
    obs_lab = {"Z0": "$Z_0$\n(weight 1)", "Z0Z1": "$Z_0Z_1$\n(2)",
               "X0X1": "$X_0X_1$\n(2)", "Z1Z2Z3": "$Z_1Z_2Z_3$\n(3)",
               "Z_all": "$Z_0\\cdots Z_4$\n(5)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    dep = brk[brk.facet == "depth"].copy()
    dep["level"] = dep["level"].astype(int)
    for m in sel:
        s = dep[dep.model == m].sort_values("level")
        axes[0].plot(s["level"], s["rmse"], "-o", color=MODEL_C[m], mec="white",
                     mew=0.7, label=NAME[m])
    axes[0].set_xlabel("Circuit depth (layers)")
    axes[0].set_ylabel("RMSE vs ideal")
    axes[0].set_xticks(sorted(dep["level"].unique()))
    ratio = (dep[dep.model == "no_mitigation"].sort_values("level")["rmse"].iloc[-1] /
             dep[dep.model == "no_mitigation"].sort_values("level")["rmse"].iloc[0])
    axes[0].set_title(f"Unmitigated error grows {ratio:.1f}$\\times$ from depth 2 to 6")

    obs = brk[brk.facet == "observable"]
    xi = np.arange(len(obs_order))
    for m in sel:
        s = obs[obs.model == m].set_index("level").loc[obs_order]
        axes[1].plot(xi, s["rmse"], "-o", color=MODEL_C[m], mec="white", mew=0.7)
    axes[1].set_xticks(xi); axes[1].set_xticklabels([obs_lab[o] for o in obs_order])
    axes[1].set_xlabel("Observable (Pauli weight)")
    axes[1].set_ylabel("RMSE vs ideal")
    axes[1].set_title("Error is nearly flat across observable weight")
    top = max(dep.rmse.max(), obs[obs.model.isin(sel)].rmse.max()) * 1.10
    for a in axes:
        a.set_ylim(0, top); a.margins(x=0.07)
    axes[0].legend(loc="upper left", handlelength=1.4, labelspacing=0.3, borderpad=0.1)
    axes[0].text(0.98, 0.02, "lower = better", transform=axes[0].transAxes,
                 fontsize=7, color="0.45", ha="right")
    for a, l in zip(axes, "ab"):
        letter(a, l)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_breakdown.png"))
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_bandwidth(bw, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    unmit = bw["unmitigated_rmse"].iloc[0]
    axes[0].axhline(unmit, color="0.45", ls=":", lw=1.1)
    axes[0].plot(bw["bandwidth"], bw["rmse"], "-o",
                 color=KIND_C["hybrid quantum-classical"], mec="white", mew=0.7)
    axes[0].fill_between(bw["bandwidth"], bw["rmse_lo"], bw["rmse_hi"],
                         color=KIND_C["hybrid quantum-classical"], alpha=0.18, lw=0)
    axes[0].set_xscale("log"); axes[0].set_xlabel("Kernel bandwidth $\\beta$")
    axes[0].set_ylabel("RMSE vs ideal")
    axes[0].set_title("Untuned bandwidth is worse than no mitigation")
    axes[0].annotate("no mitigation", xy=(bw["bandwidth"].min(), unmit),
                     xytext=(2, 4), textcoords="offset points", fontsize=7, color="0.40")
    axes[0].annotate("default $\\beta=1$", xy=(1.0, bw.set_index("bandwidth").loc[1.0, "rmse"]),
                     xytext=(-62, 12), textcoords="offset points", fontsize=7,
                     color=KIND_C["hybrid quantum-classical"],
                     arrowprops=dict(arrowstyle="-", lw=0.8,
                                     color=KIND_C["hybrid quantum-classical"]))
    axes[0].text(0.98, 0.06, "lower = better", transform=axes[0].transAxes,
                 fontsize=7, color="0.45", ha="right")

    axes[1].plot(bw["bandwidth"], bw["mean_offdiag_kernel"], "-o", color="#3f5f7f",
                 mec="white", mew=0.7, label="mean")
    axes[1].fill_between(bw["bandwidth"], bw["mean_offdiag_kernel"] - bw["std_offdiag_kernel"],
                         bw["mean_offdiag_kernel"] + bw["std_offdiag_kernel"],
                         color="#3f5f7f", alpha=0.18, lw=0, label="$\\pm$ 1 SD across entries")
    axes[1].legend(loc="lower left", fontsize=7)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Kernel bandwidth $\\beta$")
    axes[1].set_ylabel("Off-diagonal kernel entry")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Mean off-diagonal kernel value falls with bandwidth")
    for a, l in zip(axes, "ab"):
        letter(a, l)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_bandwidth.png"))
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_sweeps(sw, out):
    sel = ["no_mitigation", "zne_richardson2", "hist_gbm", "mlp",
           "rbf_krr", "qk_ridge_quantum", "bounded_vqc_quantum"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    e1 = sw[sw.experiment == "E1_noise_strength"].copy()
    e1["x"] = e1["config"].str.rstrip("x").astype(float)
    for m in sel:
        s = e1[e1.model == m].sort_values("x")
        if len(s):
            axes[0].plot(s["x"], s["rmse"], "-o", color=MODEL_C[m], mec="white",
                         mew=0.7, label=NAME[m])
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Base-noise multiplier $m$")
    axes[0].set_ylabel("RMSE vs ideal")
    multipliers = sorted(e1["x"].unique())
    axes[0].set_xticks(multipliers, labels=[f"{m:g}" for m in multipliers])
    axes[0].set_title("Paired noise-strength sweep")

    e2 = sw[sw.experiment == "E2_shot_budget"].copy()
    xmap = {"256": 256, "1024": 1024, "4096": 4096, "analytic": 16384}
    e2["x"] = e2["config"].map(xmap)
    for m in sel:
        s = e2[e2.model == m].sort_values("x")
        if len(s):
            axes[1].plot(s["x"], s["rmse"], "-o", color=MODEL_C[m], mec="white", mew=0.7)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([256, 1024, 4096, 16384])
    axes[1].set_xticklabels(["256", "1k", "4k", "analytic"])
    axes[1].set_xlabel("Shots per expectation value")
    axes[1].set_ylabel("RMSE vs ideal")
    axes[1].set_title("Input shot-budget sweep")

    e3 = sw[sw.experiment == "E3_qubit_count"].copy()
    e3["x"] = e3["config"].str.rstrip("q").astype(int)
    for m in sel:
        s = e3[e3.model == m].sort_values("x")
        if len(s):
            axes[2].plot(s["x"], s["rmse"], "-o", color=MODEL_C[m], mec="white", mew=0.7)
    axes[2].set_xticks(sorted(e3["x"].unique()))
    axes[2].set_xlabel("Number of qubits")
    axes[2].set_ylabel("RMSE vs ideal")
    axes[2].set_title("System-size sweep")

    axes[0].legend(loc="upper left", handlelength=1.4, labelspacing=0.25,
                   borderpad=0.1, fontsize=6.5)
    axes[0].text(0.98, 0.04, "lower = better", transform=axes[0].transAxes,
                 fontsize=7, color="0.45", ha="right")
    for a in axes:
        a.margins(x=0.08)
        a.set_ylim(bottom=0)
    for a, l in zip(axes, "abc"):
        letter(a, l, dx=-0.20)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_sweeps.png"))
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def fig_learning(sw, out):
    e5 = sw[sw.experiment == "E5_learning_curve"].copy()
    e5["n_circ"] = e5["extra"].str.extract(r"n_train_circuits=(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for m in ["zne_richardson2", "ridge", "hist_gbm", "mlp", "rbf_krr", "qk_ridge_quantum",
              "dr_vqc_quantum", "bounded_vqc_quantum"]:
        s = e5[e5.model == m].sort_values("n_circ")
        if not len(s):
            continue
        ls = ":" if m == "zne_richardson2" else "-"
        ax.plot(s["n_circ"], s["rmse"], ls, marker="o", color=MODEL_C[m],
                mec="white", mew=0.7, label=NAME[m])
    unmit = e5[e5.model == "no_mitigation"]["rmse"].mean()
    ax.axhline(unmit, color="0.45", ls=":", lw=1.1)
    ax.annotate("no mitigation", xy=(e5["n_circ"].min(), unmit), xytext=(40, 4),
                textcoords="offset points", fontsize=7, color="0.40")
    ax.set_xscale("log")
    ticks = sorted(e5["n_circ"].unique())
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Reference circuits used for training")
    ax.set_ylabel("RMSE vs ideal")
    ax.set_title("Error versus the number of reference circuits")
    ax.text(0.98, 0.03, "lower = better", transform=ax.transAxes, fontsize=7,
            color="0.45", ha="right")
    ax.legend(loc="upper right", handlelength=1.4, labelspacing=0.22, fontsize=6.2,
              ncol=2, columnspacing=0.9)
    ax.set_ylim(0, e5["rmse"].max() * 1.38)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_learning.png"))
    plt.close(fig)


# ------------------------------------------------------------------ figure 6
def fig_ood(sw, out):
    e4 = sw[sw.experiment == "E4_bounded_encoding"].set_index("model")
    e7 = sw[sw.experiment == "E7_cross_family"]
    sel = ["no_mitigation", "zne_richardson2", "hist_gbm", "mlp",
           "rbf_krr", "qk_ridge_quantum", "dr_vqc_quantum", "bounded_vqc_quantum"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))

    present = [m for m in sel if m in e4.index]
    y = np.arange(len(present))
    ax = axes[0]
    ax.axvline(e4.loc["no_mitigation", "rmse"], color="0.45", ls=":", lw=1.1)
    for yi, m in zip(y, present):
        c = MODEL_C[m]
        ax.hlines(yi, 0, e4.loc[m, "rmse"], color=c, lw=1.0, alpha=0.45)
        ax.plot([e4.loc[m, "rmse_lo"], e4.loc[m, "rmse_hi"]], [yi, yi], color=c,
                lw=2.6, alpha=0.45, solid_capstyle="butt")
        ax.plot(e4.loc[m, "rmse"], yi, "o", color=c, ms=6.5, mec="white", mew=0.8)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in present])
    ax.set_xlabel("RMSE vs ideal  (lower = better)")
    ax.set_title("Bounded encoding: better, still worse than nothing")
    ax.margins(y=0.07)

    ax = axes[1]
    cfgs = ["hea_to_tfim", "tfim_to_hea"]
    lab = {"hea_to_tfim": "Train HEA\ntest Ising", "tfim_to_hea": "Train Ising\ntest HEA"}
    w = 0.8 / len(present)
    for j, m in enumerate(present):
        vals = [e7[(e7.config == c) & (e7.model == m)]["rmse"].iloc[0]
                if len(e7[(e7.config == c) & (e7.model == m)]) else np.nan
                for c in cfgs]
        xs = np.arange(len(cfgs)) + j * w - 0.4 + w / 2
        ax.bar(xs, vals, width=w * 0.9, color=MODEL_C[m], label=NAME[m])
    ax.set_xticks(np.arange(len(cfgs))); ax.set_xticklabels([lab[c] for c in cfgs])
    ax.set_ylabel("RMSE vs ideal")
    ax.set_title("Transfer is asymmetric across families")
    ax.set_ylim(0, e7[e7.model.isin(present)]["rmse"].max() * 1.55)
    ax.legend(loc="upper left", handlelength=1.0, labelspacing=0.22, fontsize=6.2,
              ncol=2, columnspacing=0.8)
    for a, l in zip(axes, "ab"):
        letter(a, l, dx=-0.30)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ch5_ood.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    style()
    res = pd.read_csv("benchmark_results.csv")
    brk = pd.read_csv("breakdown_splitA.csv")
    sw = pd.read_csv("sweep_results.csv")
    bwd = pd.read_csv("bandwidth_study.csv")
    fig_main(res, a.outdir)
    fig_breakdown(brk, a.outdir)
    fig_bandwidth(bwd, a.outdir)
    fig_sweeps(sw, a.outdir)
    fig_learning(sw, a.outdir)
    fig_ood(sw, a.outdir)
    print("wrote:", sorted(f for f in os.listdir(a.outdir) if f.endswith(".png")))


if __name__ == "__main__":
    main()
