"""Regenerate the regime-II, robustness and non-Markovianity parts of Appendix B
from the result files, so the appendix cannot disagree with them.

Replaces everything in appendices/AppendixB.tex from
"\\section{Realistic regime: all experiments}" up to (not including)
"\\section{Selected hyperparameters}".

Usage: python make_appendixB.py <project_root>
"""
import json
import os
import sys

import pandas as pd

NAME = {"no_mitigation": "No mitigation", "zne_linear": "ZNE, linear",
        "zne_richardson2": "ZNE, Richardson (2)", "zne_exponential": "ZNE, exponential",
        "ridge": "Ridge", "hist_gbm": "GB trees", "mlp": "Neural net",
        "qk_ridge_quantum": "Quantum kernel ridge", "dr_vqc_quantum": "DR-VQC",
        "bounded_vqc_quantum": "Bounded VQC"}
ORDER = list(NAME)
SPLIT = {"A_random_circuits": "A", "B_depth_extrapolation": "B", "C_drift": "C"}


SHORTCAP = {'Device-realistic regime, all experiments except': 'Realistic regime, all experiments', 'Shot-limited quantum kernel (realistic regime': 'Shot-limited quantum kernel, every shot-noise draw', 'Every paired comparison in the realistic regime': 'Realistic regime, all paired comparisons', 'Ten circuit-grouped splits (seeds': 'Robustness over ten circuit-grouped splits', 'Variational arms, ten initialisation seeds': 'Variational arms over ten initialisation seeds', 'Depth extrapolation (Split B, idealized regime) by input set': 'Depth extrapolation by input set', 'Learning-curve variance: $20$': 'Learning-curve variance'}


def tex(x):
    return str(x).replace("_", "\\_")


def longtable(caption, label, cols, header, rows):
    return "{\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n" + _longtable(caption, label, cols, header, rows) + "\n}"


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
    d = sw[sw.experiment == "R4_shot_limited_kernel"].sort_values(["shots_per_entry", "draw"])
    rows = []
    for r in d.itertuples():
        sh = "exact" if r.shots_per_entry == 0 else f"{int(r.shots_per_entry):d}"
        rows.append([sh, f"{int(r.draw)}", f"{r.circuit_executions:.2e}" if r.shots_per_entry else "---",
                     f"{r.executions_per_prediction:.2e}" if r.shots_per_entry else "---",
                     f"{r.rmse:.4f}", f"{r.rmse_lo:.4f}", f"{r.rmse_hi:.4f}"])
    return longtable(
        "Shot-limited quantum kernel (realistic regime, Split A), every shot-noise draw. "
        "Executions are totals to build and apply the kernel; per prediction is one "
        "fidelity circuit per training point times the shots per entry.",
        "tab:appB_shotkernel", "rrcccccc"[:7],
        ["Shots/entry", "Draw", "Total exec.", "Per prediction", "RMSE", "CI low", "CI high"], rows)


SCALE = {"parameter_scaling": "param.", "unitary_folding": "fold."}
COND = {"markovian_control": "defect removed", "nonmarkovian_production": "production",
        "reset_control": "reset control"}
DIFF = {"arm_minus_unmitigated": "method $-$ unmitigated", "arm_minus_zne_linear": "method $-$ linear ZNE",
        "production_minus_markovian_control": "production $-$ defect removed",
        "production_minus_reset_control": "production $-$ reset control",
        "folding_minus_parameter": "fold. $-$ param.", "drift_minus_no_drift": "calib. B $-$ calib. A"}


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
        "fold. = unitary folding; defect removed, reset control and production are the three "
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
        "resolved in the stated direction. Setting: I = idealized regime; II = realistic "
        "production model; II reset = reset control; param. = parameter scaling; fold. = "
        "unitary folding.",
        "tab:appB_robust", "llccccccccc",
        ["Setting", "Method", "Mean", "SD", "Rank", "$>$unm.", "$<$unm.", "$>$lin.", "$<$lin.",
         "$>$NN", "$<$NN"], rows)


def vqc_seeds(v):
    d = v.sort_values(["split", "model", "epochs", "seed"])
    rows = [[SPLIT[r.split], NAME[r.model], f"{int(r.epochs)}", f"{int(r.seed)}", f"{r.rmse:.4f}",
             f"{r.rmse_lo:.4f}", f"{r.rmse_hi:.4f}", f"{r.err_reduction_pct:.1f}"] for r in d.itertuples()]
    return longtable(
        "Variational arms, ten initialisation seeds on the reference split (seed 7 split), "
        "idealized regime.",
        "tab:appB_vqcseeds", "llcccccr",
        ["Split", "Method", "Epochs", "Seed", "RMSE", "CI low", "CI high", "Red.\\%"], rows)


def followups(fr, fp):
    f1 = fr[fr.experiment == "F1_ood_features"].copy()
    f1["k"] = f1.model.map(order_key)
    f1 = f1.sort_values(["config", "k"])
    pp = fp.set_index(["config", "model"])
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
        "Depth extrapolation (Split B, idealized regime) by input set. The last column is the "
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
        "Split A test set, idealized regime. The last column counts draws worse than no "
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
    body = "\n\n".join([
        "\\section{Realistic regime: all experiments}\n\n" + realistic_main(sw),
        "\\section{Realistic regime: shot-limited kernel}\n\n" + shot_kernel(sw),
        "\\section{Realistic regime: paired comparisons}\n\n" + paired(pr),
        "\\section{Robustness over ten splits}\n\n" + robustness(rs),
        "\\section{Variational arms over ten initialisation seeds}\n\n" + vqc_seeds(v),
        "\\section{Input support and learning-curve variance}\n\n" + followups(fr, fp),
        "\\section{Non-Markovianity measurements}\n\n" + blp(nm, rc),
    ]) + "\n\n"
    F = os.path.join(root, "appendices", "AppendixB.tex")
    s = open(F).read()
    a = s.index("\\section{Realistic regime: all experiments}")
    b = s.index("\\section{Selected hyperparameters}")
    open(F, "w").write(s[:a] + body + s[b:])
    print("Appendix B regenerated:", F)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
