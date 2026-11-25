"""Regenerate the mitigation tables in Appendix B from the result files.

Usage: python make_appendixB.py <project_root>
"""
import json
import os
import sys

import pandas as pd

NAME = {"no_mitigation": "No mitigation", "zne_linear": "ZNE, linear",
        "zne_richardson2": "ZNE, Richardson (2)", "zne_exponential": "ZNE, exponential",
        "ridge": "Ridge", "hist_gbm": "GB trees", "mlp": "Neural net",
        "rbf_krr": "Classical RBF KRR",
        "qk_ridge_quantum": "Quantum kernel ridge", "dr_vqc_quantum": "DR-VQC",
        "bounded_vqc_quantum": "Bounded VQC"}
ORDER = list(NAME)
SPLIT = {"A_random_circuits": "A", "B_depth_extrapolation": "B", "C_drift": "C"}


SHORTCAP = {
    "Main benchmark, idealised Markovian regime": "Main benchmark, idealised regime, both splits",
    "Device-realistic regime, all experiments except": "Realistic regime, all experiments",
    "Shot-limited quantum kernel (realistic regime": "Shot-limited quantum kernel, every shot-noise draw",
    "Every paired comparison in the realistic regime": "Realistic regime, all paired comparisons",
    "Ten circuit-grouped splits (seeds": "Robustness over ten circuit-grouped splits",
    "Crossed circuit and shot seeds under realistic production noise": "Scaling contrasts over circuit and shot seeds",
    "Pool-level intervals on the realistic depth holdout": "Pool-level intervals on the depth holdout",
    "Depth-aware kernel tuning on the realistic depth holdout": "Pool-level intervals after depth-aware tuning",
    "Locked depth-aware validation on fresh pools": "Pool-level intervals on fresh validation pools",
    "Fidelity shot budgets on the locked folded depth holdout": "Pool-level intervals by fidelity shot budget",
    "Simulated acquisition cost on the locked folded depth holdout": "Folded-depth fidelity acquisition cost",
    "Depth-holdout RMSE by shot budget and fitting rule": "Fixed and retuned fidelity RMSE by budget",
    "Paired budget and ridge-penalty contrasts": "Pool-level intervals for paired budget and penalty contrasts",
    "Sensitivity of the six contrasts to the kernel-shot seed": "Budget and penalty contrasts by kernel-shot seed",
    "Fixed-parameter fidelity RMSE on fresh folded depth pools": "Fixed-parameter fidelity RMSE by budget",
    "Paired shot-budget changes at fixed bandwidth": "Pool-level intervals for fixed-parameter budget contrasts",
    "Sensitivity of the successive budget changes": "Fixed-parameter budget changes by kernel-shot seed",
    "Matched RBF minus fixed fidelity RMSE": "Matched RBF minus fixed fidelity RMSE",
    "Circuit execution counts for the matched RBF comparison": "Circuit costs for matched RBF and fixed fidelity",
    "Kernel-shot seed sensitivity of RBF minus fixed fidelity": "Matched RBF comparisons by kernel-shot seed",
    "Locked RBF minus fixed fidelity RMSE": "Locked RBF minus fixed fidelity RMSE",
    "Circuit execution counts with both kernel settings fixed": "Circuit costs with both kernel settings fixed",
    "Kernel-shot seed sensitivity with both kernels fixed": "Fixed-kernel comparisons by kernel-shot seed",
    "Fidelity RMSE and budget effects after matched selection": "Fidelity accuracy after matched earlier-pool selection",
    "Fidelity settings and one-time calibration costs": "Fidelity settings and one-time calibration costs",
    "Fidelity RMSE with budget-specific settings": "Fidelity accuracy with budget-specific settings",
    "Locked RBF minus budget-calibrated fidelity RMSE": "RBF versus budget-calibrated fidelity",
    "Circuit execution counts with budget-calibrated fidelity": "Circuit costs with budget-calibrated fidelity",
    "Kernel-shot seed sensitivity with budget-calibrated fidelity": "Budget-calibrated fidelity comparisons by seed",
    "Variational arms, ten initialisation seeds": "Variational arms over ten initialisation seeds",
    "Depth extrapolation (Split B, idealised regime) by input set": "Depth extrapolation by input set",
    "Learning-curve variance: $20$": "Learning-curve variance",
}


INCLUDE_FOLDING_DEPTH = False


def tex(x):
    return str(x).replace("_", "\\_")


def longtable(caption, label, cols, header, rows, *, colsep=4):
    preamble = "{\\footnotesize\n\\setlength{\\tabcolsep}{" + f"{colsep}pt" + "}\n"
    return preamble + _longtable(caption, label, cols, header, rows) + "\n}"


def _longtable(caption, label, cols, header, rows):
    h = " & ".join(f"\\textbf{{{c}}}" for c in header) + " \\\\"
    short = next((v for k, v in SHORTCAP.items() if caption.startswith(k)), None)
    cap = f"\\caption[{short}]{{{caption}}}" if short else f"\\caption{{{caption}}}"
    out = [f"\\begin{{longtable}}{{{cols}}}",
           cap + f"\\label{{{label}}}\\\\",
           "\\toprule", h, "\\midrule", "\\endfirsthead",
           "\\toprule", h, "\\midrule", "\\endhead",
           "\\bottomrule", "\\endfoot"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out.append("\\end{longtable}")
    return "\n".join(out)


def order_key(m):
    return ORDER.index(m) if m in ORDER else 99


def ideal_main(bm):
    rows = []
    for r in bm.sort_values(["split", "rmse"]).itertuples():
        if r.model == "qk_ridge_quantum":
            evaluations = f"3 + {int(r.n_train)}"
        elif r.model == "dr_vqc_quantum":
            evaluations = "3 + 1"
        else:
            evaluations = f"{int(r.circuit_evals_per_pred)}"
        rows.append([SPLIT[r.split], NAME[r.model], f"{r.rmse:.4f}", f"{r.rmse_lo:.4f}",
                     f"{r.rmse_hi:.4f}", f"{r.mae:.4f}", f"{r.r2:.3f}",
                     f"{r.err_reduction_pct:.1f}", evaluations, f"{int(r.n_params)}"])
    return longtable(
        "Main benchmark, idealised Markovian regime, both splits. Split A is the random "
        "held-out test, and Split B tests deeper circuits. Evals counts quantum circuits per "
        "prediction before shot multiplication. The classical RBF kernel uses the three "
        "input measurements. The fidelity kernel adds one circuit per training point, "
        "and the VQC adds one circuit (Section~\\ref{sec:ch5_cost}).",
        "tab:appB_main", "llcccccccc",
        ["Split", "Method", "RMSE", "CI low", "CI high", "MAE", "$R^2$", "Red.\\%",
         "Evals", "Params"], rows)


def ideal_sweeps(sw):
    rows = [[r.experiment.split("_", 1)[0], tex(r.config), NAME[r.model], f"{r.rmse:.4f}",
             f"{r.rmse_lo:.4f}", f"{r.rmse_hi:.4f}", f"{r.err_reduction_pct:.1f}"]
            for r in sw.sort_values(["experiment", "config", "rmse"]).itertuples()]
    return longtable(
        "All sweep experiments, idealised regime. E1 noise strength, E2 shot budget, "
        "E3 system size, E4 bounded-encoding VQC, E5 learning curve, E6 feature ablation, "
        "E7 cross-family transfer. All E1 settings use the same circuit pool and grouped "
        "holdout. Sweep points use the reduced grid and 30-epoch VQC budget of "
        "Table~\\ref{tab:ch4_grids}, with the full budget used for E3 at $n=5$ and E4.",
        "tab:appB_sweeps", "lllcccc",
        ["Experiment", "Config", "Method", "RMSE", "CI low", "CI high", "Red.\\%"], rows)


def selected_kernels(bm):
    rows = [[SPLIT[r.split], NAME[r.model], f"{r.kernel_scale:g}",
             f"{r.kernel_alpha:.2e}"] for r in bm.itertuples()
            if r.model in ("rbf_krr", "qk_ridge_quantum")]
    return longtable(
        "Selected kernel parameters in the idealised main benchmark. Spread is the "
        "RBF coefficient $\\gamma$ or the fidelity-map bandwidth $\\beta$.",
        "tab:appB_kernelparams", "llcc", ["Split", "Kernel", "Spread", "$\\alpha$"], rows)


def realistic_main(sw):
    d = sw[sw.experiment != "R4_shot_limited_kernel"].copy()
    d["k"] = d.model.map(order_key)
    d = d.sort_values(["experiment", "config", "split", "k"])
    rows = [[tex(r.experiment.split("_", 1)[0]), comp_label(r.config), SPLIT.get(r.split, tex(r.split)),
             NAME.get(r.model, tex(r.model)), f"{r.rmse:.4f}", f"{r.rmse_lo:.4f}", f"{r.rmse_hi:.4f}",
             f"{r.err_reduction_pct:.1f}"] for r in d.itertuples()]
    return longtable(
        "Device-realistic regime, all experiments except the shot-limited kernel. R1: scaling "
        "method; R2: memory, with configuration given as scaling method : noise condition "
        "(markovian\\_control = defect removed, reset\\_control = defect reset every layer, "
        "nonmarkovian\\_production = defect with memory); R3: calibration drift. Split A = "
        "held-out random circuits, B = deeper circuits, C = drifted calibration. Intervals are "
        "circuit-grouped bootstrap $95\\%$ intervals; Red.\\% is error reduction relative to the "
        "unmitigated estimator of the same configuration.",
        "tab:appB_realistic", "llllcccr",
        ["Exp.", "Configuration", "Split", "Method", "RMSE", "CI low", "CI high", "Red.\\%"], rows)


def shot_kernel(sw):
    d = sw[(sw.experiment == "R4_shot_limited_kernel") & (sw.model == "qk_ridge_quantum")]
    d = d.sort_values(["shots_per_entry", "draw"])
    rows = []
    for r in d.itertuples():
        sh = "exact" if r.shots_per_entry == 0 else f"{int(r.shots_per_entry):d}"
        costs = [r.cv_circuit_executions, r.final_circuit_executions,
                 r.executions_per_prediction]
        rows.append([sh, f"{int(r.draw)}", f"{r.kernel_alpha:.2g}",
                     *[f"{cost:.2e}" if r.shots_per_entry else "---" for cost in costs],
                     f"{r.rmse:.4f}", f"{r.rmse_lo:.4f}--{r.rmse_hi:.4f}"])
    return longtable(
        "Shot-limited quantum kernel (realistic regime, Split A), every shot-noise draw. "
        "$\\alpha$ is selected by mean grouped validation $R^2$ after fold-local "
        "PSD repair. Tuning and final executions cover their respective training and "
        "cross-kernel blocks. Per prediction is one fidelity circuit per final "
        "training point times the shots per entry.",
        "tab:appB_shotkernel", "rrrrrrrr",
        ["Shots/entry", "Draw", "$\\alpha$", "Tuning exec.", "Final exec.",
         "Per prediction", "RMSE", "95\\% CI"], rows, colsep=3)


SCALE = {"parameter_scaling": "param.", "unitary_folding": "fold."}
COND = {"markovian_control": "defect removed", "nonmarkovian_production": "production",
        "reset_control": "reset control"}
DIFF = {"arm_minus_unmitigated": "method $-$ unmitigated", "arm_minus_zne_linear": "method $-$ linear ZNE",
        "production_minus_markovian_control": "production $-$ defect removed",
        "production_minus_reset_control": "production $-$ reset control",
        "folding_minus_parameter": "fold. $-$ param.", "drift_minus_no_drift": "calib. B $-$ calib. A",
        "qk_minus_rbf": "fidelity $-$ RBF", "rbf_minus_fidelity": "RBF $-$ fidelity"}


def comp_label(c):
    parts = c.split(":")
    out = []
    for p in parts:
        out.append(SCALE.get(p) or COND.get(p) or DIFF.get(p) or tex(p))
    return ", ".join(out)


def paired(pr):
    d = pr.copy()
    d["k"] = d.model.map(order_key)
    d = d.sort_values(["experiment", "comparison", "split", "k"])
    rows = [[tex(r.experiment.split("_", 1)[0]), comp_label(r.comparison), SPLIT.get(r.split, tex(r.split)),
             NAME.get(r.model, tex(r.model)), f"{r.diff:+.4f}", f"{r.diff_lo:+.4f}", f"{r.diff_hi:+.4f}",
             "yes" if r.resolved else "no"] for r in d.itertuples()]
    return longtable(
        "Every paired comparison in the realistic regime: circuit-grouped paired bootstrap "
        "($1000$ resamples) on the RMSE difference between two conditions or methods on "
        "identical test rows. A positive difference means the first-named condition has the "
        "larger error. Resolved = the $95\\%$ interval excludes zero. param. = parameter scaling; "
        "fold. = unitary folding. The two scaling protocols share the scale-1 observations "
        "within each condition. Defect removed, reset control and production are the three "
        "noise conditions of Section~\\ref{sec:ch3_realistic}.",
        "tab:appB_paired", "l>{\\raggedright\\arraybackslash}p{4.4cm}llcccc",
        ["Exp.", "Comparison", "Split", "Method", "Diff.", "Low", "High", "Res."], rows)


def robustness(rs):
    lab = {"ideal": "I", "real_param": "II, param.", "real_fold": "II, fold.",
           "real_reset_param": "II reset, param.", "real_reset_fold": "II reset, fold."}
    d = rs.copy()
    d["k"] = d.model.map(order_key)
    d["c"] = d.config.map(list(lab).index)
    d = d.sort_values(["c", "mean"])
    rows = [[lab[r.config], NAME.get(r.model, tex(r.model)), f"{r.mean:.4f}", f"{r.sd:.4f}",
             f"{r.mean_rank:.1f}", str(r.better_than_unmit), str(r.worse_than_unmit),
             str(r.better_than_zne_lin), str(r.worse_than_zne_lin), str(r.better_than_mlp),
             str(r.worse_than_mlp)] for r in d.itertuples()]
    return longtable(
        "Ten circuit-grouped splits (seeds $100$--$109$) per configuration. Mean and SD of "
        "test RMSE over splits, mean within-split rank, and the number of splits in which the "
        "paired difference against no mitigation, linear ZNE and the neural network is "
        "resolved in the stated direction. Setting: I = idealised regime; II = realistic "
        "production model; II reset = reset control; param. = parameter scaling; fold. = "
        "unitary folding.",
        "tab:appB_robust", "llccccccccc",
        ["Setting", "Method", "Mean", "SD", "Rank", "$<$unm.", "$>$unm.", "$<$lin.", "$>$lin.",
         "$<$NN", "$>$NN"], rows, colsep=3)


def folding_seed_replication(summary):
    comparisons = {
        "parameter_scaling:rbf_minus_fidelity": r"Param. RBF $-$ fidelity",
        "unitary_folding:rbf_minus_fidelity": r"Folded RBF $-$ fidelity",
    }
    d = summary[summary.comparison.isin(["folding_minus_parameter", *comparisons])].copy()
    d["k"] = d.model.map(order_key)
    rows = []
    for r in d.sort_values(["split", "comparison", "k"]).itertuples():
        label = comparisons.get(r.comparison, NAME[r.model])
        rows.append([
            SPLIT[r.split],
            label,
            f"{r.diff_mean:+.5f}",
            f"[{r.diff_min:+.5f}, {r.diff_max:+.5f}]",
            f"[{r.pool_mean_min:+.5f}, {r.pool_mean_max:+.5f}]",
            f"{int(r.n_resolved_negative)}/{int(r.n_cells)}",
            f"{int(r.n_resolved_positive)}/{int(r.n_cells)}",
        ])
    return longtable(
        "Crossed circuit and shot seeds under realistic production noise. Named estimators "
        "report folded minus parameter-scaled RMSE. The seed-cell range is over all seed "
        "combinations. The pool-mean range is over the mean differences within each circuit "
        "pool across its shot draws. The last columns count within-cell $95\\%$ paired "
        "intervals entirely below or above zero. The cells sharing circuits are dependent.",
        "tab:appB_foldingseeds", "llccccc",
        ["Split", "Contrast", "Mean", "Seed-cell range", "Pool-mean range", "$<0$", "$>0$"],
        rows, colsep=3,
    )


def folding_depth_replication(summary, *, study="mixed_depth"):
    protocol_short = {"parameter_scaling": "Param.", "unitary_folding": "Folded"}
    short_model = {"rbf_krr": "RBF", "qk_ridge_quantum": "fidelity"}
    rows = []
    for r in summary.sort_values(["cohort", "comparison", "model"]).itertuples():
        if r.comparison == "folding_minus_parameter" and r.cohort == "all_pools":
            label = NAME[r.model]
        elif r.comparison.endswith(":depth_aware_minus_groupkfold"):
            protocol = r.comparison.split(":", 1)[0]
            label = protocol_short[protocol] + " " + short_model[r.model] + " tuning change"
        elif r.comparison.endswith(":rbf_minus_fidelity"):
            protocol = r.comparison.split(":", 1)[0]
            label = protocol_short[protocol] + r" RBF $-$ fidelity"
        elif (r.comparison.endswith(":arm_minus_zne_linear") and r.model in short_model
              and r.cohort == "all_pools"):
            protocol = r.comparison.split(":", 1)[0]
            label = protocol_short[protocol] + " " + short_model[r.model] + r" $-$ lin. ZNE"
        else:
            continue
        rows.append([
            label, str(r.n_pools), f"{r.diff_mean:+.5f}",
            f"[{r.ci95_t_lo:+.5f}, {r.ci95_t_hi:+.5f}]",
            f"[{r.ci95_boot_lo:+.5f}, {r.ci95_boot_hi:+.5f}]",
            f"[{r.loo_mean_min:+.5f}, {r.loo_mean_max:+.5f}]",
        ])
    cohort = ("Rows with $12$ pools combine the initial and additional pools, and rows with $9$ "
              "pools use the additional pools. ")
    if study == "fresh_validation":
        title = "Locked depth-aware validation on fresh pools. "
        cohort = "The estimates use only the twelve fresh pools. "
        label = "tab:appB_foldingdepth_validation"
    elif study == "depth_aware":
        title = ("Depth-aware kernel tuning on the realistic depth holdout. "
                 "Tuning-change rows report depth-aware minus mixed-depth RMSE. ")
        label = "tab:appB_foldingdepth_aware"
    else:
        title = "Pool-level intervals on the realistic depth holdout with mixed-depth tuning. "
        label = "tab:appB_foldingdepth"
    return longtable(
        title + "Standalone estimator names report folded minus parameter-scaled RMSE. "
        + cohort + "The Student $t$ and "
        "percentile pool-bootstrap intervals are nominal $95\\%$ intervals. The last column "
        "is the range of means obtained by omitting one pool at a time.",
        label, "lccccc",
        ["Contrast", "Pools", "Mean", "$t$ interval", "Bootstrap", "Leave-one-out"],
        rows, colsep=2.3,
    )


def folding_depth_shot_tables(scores, summary):
    def budget(shots):
        return f"$10^{{{len(str(shots)) - 1}}}$" if shots else "Exact"

    def interval(lo, hi):
        return f"$[{lo:+.5f},\\ {hi:+.5f}]$"

    contrast_names = {
        "rbf_minus_fidelity": "RBF $-$ fidelity",
        "fidelity_minus_exact": "Fidelity $-$ exact",
        "linear_minus_fidelity": "Linear ZNE $-$ fidelity",
    }
    rows = []
    for comparison, label in contrast_names.items():
        estimates = summary[summary.comparison == comparison].copy()
        estimates["budget_order"] = estimates.shots_per_entry.replace(0, float("inf"))
        for r in estimates.sort_values("budget_order").itertuples():
            rows.append([
                label, budget(int(r.shots_per_entry)), f"${r.diff_mean:+.5f}$",
                interval(r.ci95_t_lo, r.ci95_t_hi), interval(r.ci95_boot_lo, r.ci95_boot_hi),
                interval(r.loo_mean_min, r.loo_mean_max),
            ])
    effects = longtable(
        "Fidelity shot budgets on the locked folded depth holdout. Each estimate averages "
        "three matched draw differences within each pool and weights twelve pool means equally. "
        "The primary contrast is RBF minus fidelity at $10^5$ shots per entry. The $t$ and "
        "whole-pool bootstrap intervals are nominal pointwise $95\\%$ intervals. Leave-one-out "
        "reports the range of means after dropping one pool at a time.",
        "tab:appB_foldingdepth_shot", "lccccc",
        ["Contrast", "Shots", "Mean", "$t$ interval", "Bootstrap", "Leave-one-out"],
        rows, colsep=2.7,
    )

    def scientific(value):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"

    costs = [
        "cv_circuit_executions", "final_train_circuit_executions",
        "test_circuit_executions", "kernel_executions_per_prediction",
    ]
    finite = scores[(scores.model == "qk_ridge_quantum") & (scores.shots_per_entry > 0)]
    means = finite.groupby("shots_per_entry")[["rmse", *costs]].mean()
    rows = [[budget(shots), f"{r.rmse:.5f}", *[scientific(r[field]) for field in costs]]
            for shots, r in means.iterrows()]
    acquisitions = longtable(
        "Simulated acquisition cost on the locked folded depth holdout, averaged over the "
        "thirty-six pool and observable-shot cases at each budget. Counts include the measured "
        "training triangle and evaluation-to-training pairs. The per-prediction count uses one "
        "entry per training row. Observable-feature shots are counted separately.",
        "tab:appB_foldingdepth_shot_cost", "cccccc",
        ["Shots/entry", "RMSE", "Tuning", "Final train", "Test", "Per prediction"], rows,
    )
    return effects + "\n\n" + acquisitions


def folding_depth_shot_crossover_tables(scores, summary):
    profiles = {
        "retuned": "Retuned",
        "fixed_alpha_0.1": "Fixed $\\alpha=0.1$",
        "fixed_alpha_0.01": "Fixed $\\alpha=0.01$",
    }
    contrasts = {
        "budget_gap_retuned": "Budget, retuned",
        "budget_gap_fixed_alpha_0.1": "Budget, fixed $\\alpha=0.1$",
        "budget_gap_fixed_alpha_0.01": "Budget, fixed $\\alpha=0.01$",
        "alpha_gap_10000": "Penalty at $10^4$",
        "alpha_gap_100000": "Penalty at $10^5$",
        "retuned_gap_minus_fixed_switch_gap": "Retuned minus fixed switch",
    }

    def interval(lo, hi):
        return f"$[{lo:+.5f},\\ {hi:+.5f}]$"

    profile_means = scores.groupby(["profile", "shots_per_entry", "circuit_seed"]).rmse.mean()
    profile_means = profile_means.groupby(["profile", "shots_per_entry"]).mean()
    fit_table = longtable(
        "Depth-holdout RMSE by shot budget and fitting rule. Each row weights the twelve "
        "pool means equally, after averaging the nine observable and kernel-shot draw "
        "combinations within each pool. Both fixed fits use bandwidth $0.05$.",
        "tab:appB_foldingdepth_shot_cross_fits", "lcc",
        ["Fit", "$10^4$ shots/entry", "$10^5$ shots/entry"],
        [[label, f"{profile_means.loc[profile, 10000]:.5f}",
          f"{profile_means.loc[profile, 100000]:.5f}"] for profile, label in profiles.items()],
    )

    aggregate = summary.loc[summary.cohort == "all_kernel_seeds"].set_index("comparison")
    rows = []
    for name, label in contrasts.items():
        r = aggregate.loc[name]
        rows.append([
            label, f"${r.diff_mean:+.5f}$", interval(r.ci95_t_lo, r.ci95_t_hi),
            interval(r.ci95_boot_lo, r.ci95_boot_hi), interval(r.loo_mean_min, r.loo_mean_max),
        ])
    effects = longtable(
        "Paired budget and ridge-penalty contrasts, with equal weight per circuit pool. "
        "Budget rows report RMSE at $10^5$ minus RMSE at $10^4$. Penalty rows report "
        "fixed $\\alpha=0.01$ minus fixed $\\alpha=0.1$. The fixed switch is $10^5$ at "
        "$\\alpha=0.01$ minus $10^4$ at $\\alpha=0.1$. "
        "The retuned budget gap is primary. The circuit-pool $t$ and bootstrap intervals "
        "are nominal pointwise $95\\%$ intervals. Leave-one-out gives the range of means "
        "after omitting each pool in turn.",
        "tab:appB_foldingdepth_shot_cross", "lcccc",
        ["Contrast", "Mean", "$t$ interval", "Bootstrap", "Leave-one-out"], rows, colsep=2.6,
    )

    by_seed = summary.loc[summary.cohort == "kernel_seed"].set_index(["comparison", "kernel_shot_seed"])
    seeds = sorted(by_seed.index.get_level_values("kernel_shot_seed").unique())
    rows = []
    for name, label in contrasts.items():
        row = [label]
        for seed in seeds:
            r = by_seed.loc[name, seed]
            row.append("\\begin{tabular}{@{}c@{}}"
                       f"${r.diff_mean:+.5f}$\\\\{interval(r.ci95_t_lo, r.ci95_t_hi)}"
                       "\\end{tabular}")
        rows.append(row)
    seed_table = longtable(
        "Sensitivity of the six contrasts to the kernel-shot seed. Cells show the mean "
        "and nominal pointwise $95\\%$ pool-level $t$ interval conditional on that kernel-shot "
        "seed. Each estimate averages three observable draws per pool and weights the "
        "same twelve pools equally.",
        "tab:appB_foldingdepth_shot_cross_seeds", "l" + "c" * len(seeds),
        ["Contrast", *[str(int(seed)) for seed in seeds]], rows, colsep=2.4,
    )
    return "\n\n".join((fit_table, effects, seed_table))


def folding_depth_shot_fixed_tables(summary):
    budgets = {100: "$10^2$", 1000: "$10^3$", 10000: "$10^4$", 100000: "$10^5$"}
    absolute = {f"absolute_rmse_{budget}": label for budget, label in budgets.items()}
    absolute["absolute_rmse_exact"] = "Exact entries"
    adjacent = {
        f"budget_gap_{hi}_minus_{lo}": f"{budgets[hi]} $-$ {budgets[lo]}"
        for lo, hi in zip(budgets, list(budgets)[1:])
    }
    contrasts = {
        **adjacent, **{f"finite_minus_exact_{budget}": f"{label} $-$ exact"
                       for budget, label in budgets.items()},
    }
    aggregate = summary.loc[summary.cohort == "all_kernel_seeds"].set_index("comparison")

    def interval(lo, hi, precision):
        return f"$[{lo:{precision}},\\ {hi:{precision}}]$"

    def result_rows(names, precision):
        rows = []
        for name, label in names.items():
            r = aggregate.loc[name]
            rows.append([
                label, f"${r.diff_mean:{precision}}$", interval(r.ci95_t_lo, r.ci95_t_hi, precision),
                interval(r.ci95_boot_lo, r.ci95_boot_hi, precision),
                interval(r.loo_mean_min, r.loo_mean_max, precision),
            ])
        return rows

    interval_headers = ["Mean", "$t$ interval", "Bootstrap", "Leave-one-out"]
    fit_table = longtable(
        "Fixed-parameter fidelity RMSE on fresh folded depth pools, with bandwidth $0.05$ and "
        "$\\alpha=0.1$. Means weight twelve circuit pools equally. Each pool averages nine "
        "observable and kernel-shot draw combinations for finite budgets and three observable "
        "draws for exact entries. Intervals are nominal pointwise $95\\%$ intervals. "
        "Leave-one-out is the range of means after omitting each pool in turn.",
        "tab:appB_foldingdepth_shot_fixed_fits", "lcccc",
        ["Shots/entry", *interval_headers], result_rows(absolute, ".5f"), colsep=2.6,
    )
    effects = longtable(
        "Paired shot-budget changes at fixed bandwidth $0.05$ and $\\alpha=0.1$. "
        "Each row averages matched draws before weighting the twelve circuit pools equally. "
        "The $10^5$ minus $10^4$ contrast is primary. Negative successive differences favour "
        "the higher budget. Exact-entry controls use identical hyperparameters and features. "
        "The $t$ and whole-pool bootstrap intervals are nominal pointwise $95\\%$ intervals.",
        "tab:appB_foldingdepth_shot_fixed_steps", "lcccc",
        ["Contrast", *interval_headers], result_rows(contrasts, "+.5f"), colsep=2.6,
    )

    by_seed = summary.loc[summary.cohort == "kernel_seed"].set_index(["comparison", "kernel_shot_seed"])
    seeds = sorted(by_seed.index.get_level_values("kernel_shot_seed").unique())
    seed_rows = []
    for name, label in adjacent.items():
        row = [label]
        for seed in seeds:
            r = by_seed.loc[name, seed]
            row.append("\\begin{tabular}{@{}c@{}}"
                       f"${r.diff_mean:+.5f}$\\\\ {interval(r.ci95_t_lo, r.ci95_t_hi, '+.5f')}"
                       "\\end{tabular}")
        seed_rows.append(row)
    seed_table = longtable(
        "Sensitivity of the successive budget changes to the kernel-shot seed. Cells show the "
        "paired mean and nominal pointwise $95\\%$ pool-level $t$ interval. Each seed "
        "reuses the same twelve pools and three observable draws per pool.",
        "tab:appB_foldingdepth_shot_fixed_seeds", "l" + "c" * len(seeds),
        ["Contrast", *[str(int(seed)) for seed in seeds]], seed_rows, colsep=2.4,
    )
    return "\n\n".join((fit_table, effects, seed_table))


def folding_depth_shot_fixed_rbf_tables(summary, costs, *, study="fixed_rbf"):
    budgets = {100: "$10^2$", 1000: "$10^3$", 10000: "$10^4$", 100000: "$10^5$"}
    comparisons = {f"rbf_minus_fixed_fidelity_{budget}": label for budget, label in budgets.items()}
    comparisons["rbf_minus_fixed_fidelity_exact"] = "Exact entries"
    aggregate = summary.loc[summary.cohort == "all_kernel_seeds"].set_index("comparison")

    def interval(lo, hi, precision="+.5f"):
        return f"$[{lo:{precision}},\\ {hi:{precision}}]$"

    rbf = aggregate.loc["absolute_rmse_rbf"]
    both_fixed = study != "fixed_rbf"
    label_prefix = f"tab:appB_foldingdepth_shot_{study}"
    comparison_name = "Locked RBF" if both_fixed else "Matched RBF"
    fidelity_setting = "budget-calibrated" if study == "budget_calibrated" else "fixed"
    effect_rows = []
    for name, label in comparisons.items():
        r = aggregate.loc[name]
        effect_rows.append([
            label, f"${r.diff_mean:+.5f}$", interval(r.ci95_t_lo, r.ci95_t_hi),
            interval(r.ci95_boot_lo, r.ci95_boot_hi), interval(r.loo_mean_min, r.loo_mean_max),
        ])
    effects = longtable(
        f"{comparison_name} minus {fidelity_setting} fidelity RMSE on twelve folded depth pools. "
        "Mean RBF RMSE is "
        f"${rbf.diff_mean:.5f}$ with nominal $95\\%$ pool-level $t$ interval "
        f"{interval(rbf.ci95_t_lo, rbf.ci95_t_hi, '.5f')}. The $10^5$-shot row is primary. "
        "Negative differences favour RBF. The $t$ and whole-pool bootstrap intervals are nominal "
        "pointwise $95\\%$ intervals. Leave-one-out is the range of means after omitting each pool in turn.",
        label_prefix, "lcccc",
        ["Fidelity shots/entry", "Mean", "$t$ interval", "Bootstrap", "Leave-one-out"],
        effect_rows, colsep=2.6,
    )

    def count(value):
        if value == 0:
            return "$0$"
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"

    costs = costs.set_index(["model", "shots_per_entry"])
    cost_columns = [
        "fidelity_executions_train_mean", "fidelity_executions_heldout_mean",
        "measured_executions_train_heldout_mean", "fidelity_executions_new_row_mean",
        "measured_executions_new_row_mean",
    ]
    methods = {("rbf_krr", 0): ("Locked RBF" if both_fixed else "RBF"),
               **{("qk_ridge_quantum", b): f"Fidelity {label}" for b, label in budgets.items()}}
    cost_rows = [[label, *[count(costs.loc[method, field]) for field in cost_columns]]
                 for method, label in methods.items()]
    cost_title = ("Circuit execution counts with both kernel settings fixed" if both_fixed
                  else "Circuit execution counts for the matched RBF comparison")
    if study == "budget_calibrated":
        cost_title = "Circuit execution counts with budget-calibrated fidelity"
    tuning = {
        "fixed_rbf": "Tuning RBF reuses existing features and fixed fidelity has zero kernel-tuning executions.",
        "fixed_both": ("The fresh fits use zero tuning circuit executions. The earlier selection reuses the "
                       "inputs and evaluates RBF similarities classically."),
        "recalibrated": ("The fresh fits use zero tuning circuit executions. The earlier selection reuses "
                         "the inputs and computes exact fidelity overlaps in the simulator."),
        "budget_calibrated": ("The fresh fits use zero tuning circuit executions. The one-time earlier "
                              "finite-kernel calibration costs are reported separately."),
    }[study]
    acquisitions = longtable(
        f"{cost_title}, averaged across twelve circuit "
        "pools. The train-plus-holdout total includes $5.5296\\times10^7$ shared feature "
        "executions. Each new circuit--observable row uses $12{,}288$ feature executions. "
        f"Kernel columns give additional fidelity measurements. {tuning} Exact-entry fidelity "
        "represents an analytic simulation reference.",
        label_prefix + "_cost", "lccccc",
        ["Method", "Train kernel", "Test kernel", "Train/test total", "New-row kernel", "New-row total"],
        cost_rows, colsep=2.7,
    )

    by_seed = summary.loc[summary.cohort == "kernel_seed"].set_index(["comparison", "kernel_shot_seed"])
    seeds = sorted(by_seed.index.get_level_values("kernel_shot_seed").unique())
    seed_rows = []
    for budget, label in budgets.items():
        row = [label]
        for seed in seeds:
            r = by_seed.loc[f"rbf_minus_fixed_fidelity_{budget}", seed]
            row.append("\\begin{tabular}{@{}c@{}}"
                       f"${r.diff_mean:+.5f}$\\\\ {interval(r.ci95_t_lo, r.ci95_t_hi)}"
                       "\\end{tabular}")
        seed_rows.append(row)
    seed_title = ("Kernel-shot seed sensitivity with both kernels fixed"
                  if both_fixed else "Kernel-shot seed sensitivity of RBF minus fixed fidelity RMSE")
    if study == "budget_calibrated":
        seed_title = "Kernel-shot seed sensitivity with budget-calibrated fidelity"
    seed_table = longtable(
        f"{seed_title}. Cells show the paired mean "
        "and nominal pointwise $95\\%$ pool-level $t$ interval. Each seed uses the same twelve "
        "pools and three observable draws per pool.",
        label_prefix + "_seeds", "l" + "c" * len(seeds),
        ["Shots/entry", *[str(int(seed)) for seed in seeds]], seed_rows, colsep=2.4,
    )
    separation = "\n\n\\clearpage\n\n" if study in {"recalibrated", "budget_calibrated"} else "\n\n"
    return effects + separation + "\n\n".join((acquisitions, seed_table))


def recalibrated_fidelity_table(summary, *, study="recalibrated"):
    budgets = {100: "$10^2$", 1000: "$10^3$", 10000: "$10^4$", 100000: "$10^5$"}
    budget_list = list(budgets)
    absolute = {f"absolute_rmse_{budget}": f"RMSE at {label}" for budget, label in budgets.items()}
    absolute["absolute_rmse_exact"] = "Exact-entry RMSE"
    changes = {f"budget_gap_{hi}_minus_{lo}": f"{budgets[hi]} $-$ {budgets[lo]}"
               for lo, hi in zip(budget_list, budget_list[1:])}
    gaps = {f"finite_minus_exact_{budget}": f"{label} $-$ exact" for budget, label in budgets.items()}
    aggregate = summary.loc[summary.cohort == "all_kernel_seeds"].set_index("comparison")

    rows = []
    for name, label in {**absolute, **changes, **gaps}.items():
        estimate = aggregate.loc[name]
        precision = ".5f" if name in absolute else "+.5f"
        rows.append([
            label, f"${estimate.diff_mean:{precision}}$",
            f"$[{estimate.ci95_t_lo:{precision}},\\ {estimate.ci95_t_hi:{precision}}]$",
            f"$[{estimate.ci95_boot_lo:{precision}},\\ {estimate.ci95_boot_hi:{precision}}]$",
            f"$[{estimate.loo_mean_min:{precision}},\\ {estimate.loo_mean_max:{precision}}]$",
        ])
    scope = (
        "Fidelity RMSE with budget-specific settings. Successive and finite-minus-exact "
        "rows compare the fitted procedures under those conditions. "
        if study == "budget_calibrated" else
        "Fidelity RMSE and budget effects after matched selection on the earlier pools. "
        "Successive contrasts subtract the lower-budget error from the higher-budget error. "
        "Finite-minus-exact rows use the locked exact reference. "
    )
    return longtable(
        scope + "The $t$ and whole-pool bootstrap "
        "intervals are nominal pointwise $95\\%$ intervals. Leave-one-out gives the range of means "
        "after omitting each pool in turn.",
        f"tab:appB_foldingdepth_shot_{study}_fidelity", "lcccc",
        ["Fidelity estimate", "Mean", "$t$ interval", "Bootstrap", "Leave-one-out"], rows, colsep=2.6,
    )


def budget_calibration_table(calibration, costs):
    totals = costs.groupby("shots_per_entry").sum(numeric_only=True)
    selected = calibration["selected_by_budget"]

    def count(value):
        mantissa, exponent = f"{value:.3e}".split("e")
        return "$" + f"{mantissa}\\times10^{{{int(exponent)}}}$"

    rows = []
    for exponent in range(2, 6):
        budget = 10 ** exponent
        profile = selected[str(budget)]
        measured = totals.loc[budget]
        rows.append([
            f"$10^{exponent}$", f"{profile['bandwidth']:g}", f"{profile['alpha']:g}",
            f"{profile['selection_mean_r2']:.5f}", count(measured.cv_train_circuit_executions),
            count(measured.cv_validation_circuit_executions), count(measured.cv_circuit_executions),
        ])
    feature_cost = count(costs.feature_circuit_executions.sum())
    kernel_cost = count(costs.cv_circuit_executions.sum())
    return longtable(
        "Fidelity settings and one-time calibration costs on the thirty-six earlier pool--draw "
        "cases. Each bandwidth and validation depth shares its measured training Gram and cross "
        f"entries across all ridge penalties. Kernel executions sum to {kernel_cost} across "
        f"the four budgets. Shared observed features add {feature_cost} executions once. "
        "These totals describe the modeled acquisitions underlying the retained calibration data.",
        "tab:appB_foldingdepth_shot_budget_calibrated_selection", "lcccccc",
        ["Shots/entry", "Bandwidth", r"$\alpha$", r"Mean $R^2$", "Train Gram", "Val. cross", "Kernel total"],
        rows, colsep=2.6,
    )


def vqc_seeds(v):
    d = v.sort_values(["split", "model", "epochs", "seed"])
    rows = [[SPLIT[r.split], NAME[r.model], f"{int(r.epochs)}", f"{int(r.seed)}", f"{r.rmse:.4f}",
             f"{r.rmse_lo:.4f}", f"{r.rmse_hi:.4f}", f"{r.err_reduction_pct:.1f}"] for r in d.itertuples()]
    return longtable(
        "Variational arms, ten initialisation seeds on the reference split (seed 7 split), "
        "idealised regime.",
        "tab:appB_vqcseeds", "llcccccr",
        ["Split", "Method", "Epochs", "Seed", "RMSE", "CI low", "CI high", "Red.\\%"], rows)


def followups(fr, fp):
    f1 = fr[fr.experiment == "F1_ood_features"].copy()
    f1["k"] = f1.model.map(order_key)
    f1 = f1.sort_values(["config", "k"])
    pp = fp[fp.comparison == "arm_minus_unmitigated"].set_index(["config", "model"])
    rows = []
    for r in f1.itertuples():
        if (r.config, r.model) in pp.index:
            q = pp.loc[(r.config, r.model)]
            dd = f"{q['diff']:+.4f} [{q.diff_lo:+.4f}, {q.diff_hi:+.4f}]"
        else:
            dd = "---"
        rows.append([tex(r.config), NAME.get(r.model, tex(r.model)), f"{r.rmse:.4f}",
                     f"{r.rmse_lo:.4f}", f"{r.rmse_hi:.4f}", dd])
    t1 = longtable(
        "Depth extrapolation (Split B, idealised regime) by input set. The last column is the "
        "paired difference against no mitigation (negative = better).",
        "tab:appB_oodfeat", "llcccc",
        ["Inputs", "Method", "RMSE", "CI low", "CI high", "Paired diff. vs unmitigated"], rows)
    f2 = fr[fr.experiment == "F2_learning_variance"]
    g = (f2.groupby(["config", "model"])
         .agg(mean=("rmse", "mean"), sd=("rmse", "std"), mn=("rmse", "min"), mx=("rmse", "max"),
              nw=("err_reduction_pct", lambda x: int((x < 0).sum())), n=("rmse", "size")).reset_index())
    g["k"] = g.model.map(order_key)
    g = g.sort_values(["config", "k"])
    rows = [[r.config, NAME.get(r.model, tex(r.model)), f"{r.mean:.4f}", f"{r.sd:.4f}", f"{r.mn:.4f}",
             f"{r.mx:.4f}", f"{r.nw}/{r.n}"] for r in g.itertuples()]
    t2 = longtable(
        "Learning-curve variance: $20$ independent subsets of training circuits at each size, "
        "Split A test set, idealised regime. The last column counts draws worse than no "
        "mitigation.",
        "tab:appB_learnvar", "clccccc",
        ["Circuits", "Method", "Mean", "SD", "Min", "Max", "Worse"], rows)
    return t1 + "\n\n" + t2


def blp(nm, rc):
    p = nm.get("production", nm)
    lines = ["Breuer--Laine--Piilo measure $\\mathcal{N}$ (Eq.~\\eqref{eq:ch3_blp}) on the idling "
             "register. First-layer trace distance in brackets.", "", "\\begin{itemize}"]
    for lab, key in [("non-Markovian production model", "production"),
                     ("reset control (defect reset every layer)", "reset_control"),
                     ("defect-removed control", "markovian_control")]:
        for probe, v in rc[key].items():
            lines.append(f"  \\item {lab}, probe pair {probe.split(' vs')[0]} vs. its partner: "
                         f"$\\mathcal{{N}} = {v['blp']:.4f}$ (first layer ${v['trace_distance'][0]:.3f}$)")
    sweep = nm.get("coupling_sweep")
    if sweep:
        pts = ", ".join(f"({s['g_tls']:.2f}, {s['blp_max']:.4f})" for s in sweep)
        lines.append(f"  \\item coupling sweep ($g$, $\\mathcal{{N}}$): {pts}")
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def main(root):
    R = os.path.join(root, "results")
    bm = pd.read_csv(os.path.join(R, "benchmark_results.csv"))
    ideal_sw = pd.read_csv(os.path.join(R, "sweep_results.csv"))
    sw = pd.read_csv(os.path.join(R, "sweep_realistic.csv"))
    pr = pd.read_csv(os.path.join(R, "paired_realistic.csv"))
    rs = pd.read_csv(os.path.join(R, "robustness_summary.csv"))
    v = pd.read_csv(os.path.join(R, "vqc_seed_sweep.csv"))
    fr = pd.read_csv(os.path.join(R, "followup_results.csv"))
    fp = pd.read_csv(os.path.join(R, "followup_paired.csv"))
    nm = json.load(open(os.path.join(R, "nonmarkovianity_check.json")))
    # per-probe BLP results for production and both controls, all written by
    # nonmarkov_check.py into nonmarkovianity_check.json
    rc = {"production": nm["nonmarkovian_production"]["pairs"],
          "reset_control": nm["reset_control"]["pairs"],
          "markovian_control": nm["markovian_control"]["pairs"]}
    sections = [
        "\\section{Idealised regime: main benchmark}\n\n" + ideal_main(bm),
        "\\section{Idealised regime: sweep experiments}\n\n" + ideal_sweeps(ideal_sw),
        "\\section{Realistic regime: all experiments}\n\n" + realistic_main(sw),
        "\\section{Realistic regime: shot-limited kernel}\n\n" + shot_kernel(sw),
        "\\section{Realistic regime: paired comparisons}\n\n" + paired(pr),
        "\\section{Robustness over ten splits}\n\n" + robustness(rs),
        "\\section{Variational arms over ten initialisation seeds}\n\n" + vqc_seeds(v),
        "\\section{Input support and learning-curve variance}\n\n" + followups(fr, fp),
        "\\section{Non-Markovianity measurements}\n\n" + blp(nm, rc),
        "\\section{Selected hyperparameters}\n\n" + selected_kernels(bm),
    ]
    folding_seed_path = os.path.join(R, "folding_seed_summary.csv")
    folding_depth_path = os.path.join(R, "folding_depth_summary.csv")
    folding_depth_aware_path = os.path.join(R, "folding_depth_aware_summary.csv")
    folding_depth_validation_path = os.path.join(R, "folding_depth_validation_summary.csv")
    folding_depth_shot_path = os.path.join(R, "folding_depth_shot_summary.csv")
    folding_depth_shot_crossover_path = os.path.join(R, "folding_depth_shot_crossover_summary.csv")
    folding_depth_shot_fixed_path = os.path.join(R, "folding_depth_shot_fixed_summary.csv")
    folding_depth_shot_fixed_rbf_path = os.path.join(R, "folding_depth_shot_fixed_rbf_summary.csv")
    folding_depth_shot_fixed_both_path = os.path.join(R, "folding_depth_shot_fixed_both_summary.csv")
    folding_depth_shot_recalibrated_path = os.path.join(R, "folding_depth_shot_recalibrated_summary.csv")
    budget_calibrated_path = os.path.join(R, "folding_depth_shot_budget_calibrated_summary.csv")
    if not INCLUDE_FOLDING_DEPTH:
        # the folding-depth follow-up series is not part of the thesis; skip its tables
        (folding_depth_path, folding_depth_aware_path, folding_depth_validation_path,
         folding_depth_shot_path, folding_depth_shot_crossover_path, folding_depth_shot_fixed_path,
         folding_depth_shot_fixed_rbf_path, folding_depth_shot_fixed_both_path,
         folding_depth_shot_recalibrated_path, budget_calibrated_path) = ("",) * 10
    if os.path.exists(budget_calibrated_path):
        summary = pd.read_csv(budget_calibrated_path)
        calibration_stem = os.path.join(R, "folding_depth_shot_budget_calibrated_calibration")
        with open(calibration_stem + ".json") as source:
            calibration = json.load(source)
        sections.insert(6, "\\clearpage\n\\section{Shot-budget calibration and costs}\n\n"
                        + budget_calibration_table(calibration, pd.read_csv(calibration_stem + "_costs.csv"))
                        + "\n\n" + recalibrated_fidelity_table(summary, study="budget_calibrated") + "\n\n"
                        + folding_depth_shot_fixed_rbf_tables(
                            summary,
                            pd.read_csv(os.path.join(R, "folding_depth_shot_budget_calibrated_cost_summary.csv")),
                            study="budget_calibrated",
                        ) + "\n\n\\clearpage")
    if os.path.exists(folding_depth_shot_recalibrated_path):
        summary = pd.read_csv(folding_depth_shot_recalibrated_path)
        sections.insert(6, "\\clearpage\n\\section{Matched selection and circuit cost}\n\n"
                        + recalibrated_fidelity_table(summary) + "\n\n"
                        + folding_depth_shot_fixed_rbf_tables(
                            summary,
                            pd.read_csv(os.path.join(R, "folding_depth_shot_recalibrated_cost_summary.csv")),
                            study="recalibrated",
                        ) + "\n\n\\clearpage")
    if os.path.exists(folding_depth_shot_fixed_both_path):
        sections.insert(6, "\\clearpage\n\\section{RMSE and circuit cost with both kernels fixed}\n\n"
                        + folding_depth_shot_fixed_rbf_tables(
                            pd.read_csv(folding_depth_shot_fixed_both_path),
                            pd.read_csv(os.path.join(R, "folding_depth_shot_fixed_both_cost_summary.csv")),
                            study="fixed_both",
                        ) + "\n\n\\clearpage")
    if os.path.exists(folding_depth_shot_fixed_rbf_path):
        sections.insert(6, "\\clearpage\n\\section{Matched RBF RMSE and circuit cost}\n\n"
                        + folding_depth_shot_fixed_rbf_tables(
                            pd.read_csv(folding_depth_shot_fixed_rbf_path),
                            pd.read_csv(os.path.join(R, "folding_depth_shot_fixed_rbf_cost_summary.csv")),
                        ) + "\n\n\\clearpage")
    if os.path.exists(folding_depth_shot_fixed_path):
        sections.insert(6, "\\clearpage\n\\section{Fixed-parameter fidelity shot budgets on fresh pools}\n\n"
                        + folding_depth_shot_fixed_tables(pd.read_csv(folding_depth_shot_fixed_path))
                        + "\n\n\\clearpage")
    if os.path.exists(folding_depth_shot_crossover_path):
        sections.insert(6, "\\clearpage\n\\section{Shot budget and ridge penalty on folded circuits}\n\n"
                        + folding_depth_shot_crossover_tables(
                            pd.read_csv(os.path.join(R, "folding_depth_shot_crossover_scores.csv")),
                            pd.read_csv(folding_depth_shot_crossover_path),
                        ) + "\n\n\\clearpage")
    if os.path.exists(folding_depth_shot_path):
        sections.insert(6, "\\clearpage\n\\section{Folded depth-holdout fidelity shot budgets}\n\n"
                        + folding_depth_shot_tables(
                            pd.read_csv(os.path.join(R, "folding_depth_shot_scores.csv")),
                            pd.read_csv(folding_depth_shot_path),
                        ))
    if os.path.exists(folding_depth_validation_path):
        sections.insert(6, "\\clearpage\n\\section{Locked depth-aware validation on fresh pools}\n\n"
                        + folding_depth_replication(pd.read_csv(folding_depth_validation_path),
                                                    study="fresh_validation"))
    if os.path.exists(folding_depth_aware_path):
        sections.insert(6, "\\clearpage\n\\section{Depth-aware tuning on deeper circuits}\n\n"
                        + folding_depth_replication(pd.read_csv(folding_depth_aware_path), study="depth_aware"))
    if os.path.exists(folding_depth_path):
        sections.insert(6, "\\section{Pool-level intervals on deeper circuits}\n\n"
                        + folding_depth_replication(pd.read_csv(folding_depth_path)))
    if os.path.exists(folding_seed_path):
        sections.insert(6, "\\section{Folding replication over circuit pools and shot draws}\n\n"
                        + folding_seed_replication(pd.read_csv(folding_seed_path)))
    body = "\n\n".join(sections) + "\n\n"
    F = os.path.join(root, "appendices", "AppendixB.tex")
    s = open(F).read()
    a = s.index("\\section{Idealised regime: main benchmark}")
    open(F, "w").write(s[:a] + body.rstrip() + "\n")
    print("Appendix B regenerated:", F)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
