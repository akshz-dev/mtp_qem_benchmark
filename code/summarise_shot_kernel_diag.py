"""Summarise the shot-limited-kernel diagnosis (thesis Section 5.17).

Reads the raw outputs of diag_shotkernel_alpha.py (diag_shotkernel.csv) and
diag_shotkernel_finegrid.py (diag_finegrid.csv) from the working directory and
writes shot_kernel_alpha_diagnosis.csv: per shot budget and setting, the
selected ridge penalty(ies), mean/min/max test RMSE over five shot-noise draws
and the mean most-negative eigenvalue of the shot-estimated Gram matrix.
"""
import pandas as pd

LABELS = {
    "cv_alpha_psd": "12-point alpha grid, CV per draw (thesis setting)",
    "fixed_alpha_psd": "alpha fixed at exact-Gram CV value, PSD projection",
    "fixed_alpha_noproj": "alpha fixed at exact-Gram CV value, no projection",
}


def _row(M, setting, alphas, r, eig):
    return dict(shots_per_entry=M, setting=setting,
                alpha_chosen=";".join(f"{x:.3g}" for x in sorted(set(alphas))),
                rmse_mean=r.mean(), rmse_min=r.min(), rmse_max=r.max(),
                n_draws=len(r), min_eigenvalue_mean=eig.mean())


def main():
    a = pd.read_csv("diag_shotkernel.csv")
    b = pd.read_csv("diag_finegrid.csv")
    exact = a[a.arm == "exact"].iloc[0]
    a = a[a.arm != "exact"]
    rows = []
    for (M, arm), h in a.groupby(["shots", "arm"]):
        rows.append(_row(M, LABELS[arm], h.alpha, h.rmse, h.min_eig_raw))
    for M, h in b.groupby("shots"):
        rows.append(_row(M, "25-point alpha grid, CV per draw", h.cv_alpha, h.rmse_cv_fine, h.min_eig))
        rows.append(_row(M, "25-point alpha grid, test-optimal alpha (diagnostic bound only)",
                         h.oracle_alpha, h.rmse_oracle, h.min_eig))
    out = pd.DataFrame(rows).sort_values(["setting", "shots_per_entry"])
    out["exact_gram_rmse"] = exact.rmse
    out["exact_gram_alpha"] = exact.alpha
    out.to_csv("shot_kernel_alpha_diagnosis.csv", index=False)
    print(out[["setting", "shots_per_entry", "rmse_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
