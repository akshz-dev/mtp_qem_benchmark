"""Summarise matched ridge-grid sensitivity from the corrected R4 shot draws."""

import argparse

import pandas as pd


def _chosen_alphas(alphas):
    return ";".join(f"{x:.8g}" for x in sorted(alphas.unique()))


def summarise(table):
    """Compare the two prespecified CV grids on each shared measurement draw."""
    shot = table[table.experiment.eq("R4_shot_limited_kernel")].copy()
    required = [
        "shot_protocol", "shots_per_entry", "draw", "kernel_alpha", "kernel_cv_r2",
        "kernel_candidates", "rmse", "fine_alpha", "fine_cv_r2", "fine_rmse",
        "fine_candidates", "min_eigenvalue",
    ]
    if shot.empty or not set(required).issubset(shot.columns) or shot[required].isna().any().any():
        raise ValueError("R4 rows must include the corrected shot protocol and both alpha-grid outcomes")
    if shot.shot_protocol.nunique() != 1 or shot.duplicated(["shots_per_entry", "draw"]).any():
        raise ValueError("R4 rows must use one shot protocol with a unique row per budget and draw")

    shot["shots_per_entry"] = shot.shots_per_entry.astype(int)
    shot["fine_minus_default_rmse"] = shot.fine_rmse - shot.rmse
    return shot.groupby("shots_per_entry", as_index=False).agg(
        shot_protocol=("shot_protocol", "first"),
        n_draws=("draw", "size"),
        default_candidates=("kernel_candidates", "first"),
        fine_candidates=("fine_candidates", "first"),
        default_alpha_chosen=("kernel_alpha", _chosen_alphas),
        fine_alpha_chosen=("fine_alpha", _chosen_alphas),
        default_cv_r2_mean=("kernel_cv_r2", "mean"),
        fine_cv_r2_mean=("fine_cv_r2", "mean"),
        default_rmse_mean=("rmse", "mean"),
        default_rmse_min=("rmse", "min"),
        default_rmse_max=("rmse", "max"),
        fine_rmse_mean=("fine_rmse", "mean"),
        fine_rmse_min=("fine_rmse", "min"),
        fine_rmse_max=("fine_rmse", "max"),
        fine_minus_default_rmse_mean=("fine_minus_default_rmse", "mean"),
        fine_minus_default_rmse_min=("fine_minus_default_rmse", "min"),
        fine_minus_default_rmse_max=("fine_minus_default_rmse", "max"),
        min_eigenvalue_mean=("min_eigenvalue", "mean"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="sweep_realistic.csv")
    parser.add_argument("--out", default="shot_kernel_alpha_diagnosis.csv")
    args = parser.parse_args()

    summary = summarise(pd.read_csv(args.source))
    summary.to_csv(args.out, index=False)
    display = ["shots_per_entry", "n_draws", "default_rmse_mean", "fine_rmse_mean",
               "fine_minus_default_rmse_mean"]
    print(summary[display].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
