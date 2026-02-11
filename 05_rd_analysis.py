"""
05_rd_analysis.py
Main regression discontinuity estimates using rdrobust.
"""

import os
import pandas as pd
import numpy as np
from rdrobust import rdrobust

FINAL_DIR = os.path.join(os.path.dirname(__file__), "data", "final")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "output", "tables")


def run_rd(Y, X, label, c=0):
    """Run rdrobust and return results as a dict."""
    # Drop missing — convert to numpy arrays for rdrobust compatibility
    Y = pd.Series(Y).reset_index(drop=True)
    X = pd.Series(X).reset_index(drop=True)
    mask = Y.notna() & X.notna()
    Y_clean = Y[mask].to_numpy().astype(float)
    X_clean = X[mask].to_numpy().astype(float)

    if len(Y_clean) < 50:
        print(f"  {label}: Too few observations ({len(Y_clean)}), skipping")
        return None

    try:
        result = rdrobust(Y_clean, X_clean, c=c)

        # Extract key results using .iloc[row, col] to get scalars
        coef = result.coef.iloc[0, 0]       # Conventional estimate
        se = result.se.iloc[0, 0]
        pval = result.pv.iloc[0, 0]
        ci_lower = result.ci.iloc[0, 0]
        ci_upper = result.ci.iloc[0, 1]
        bw = result.bws.iloc[0, 0]          # MSE-optimal bandwidth
        n_left = int(result.N_h[0])
        n_right = int(result.N_h[1])

        # Bias-corrected estimate
        coef_bc = result.coef.iloc[1, 0]
        se_bc = result.se.iloc[1, 0]
        pval_bc = result.pv.iloc[1, 0]

        # Robust estimate
        coef_rb = result.coef.iloc[2, 0]
        se_rb = result.se.iloc[2, 0]
        pval_rb = result.pv.iloc[2, 0]
        ci_rb_lower = result.ci.iloc[2, 0]
        ci_rb_upper = result.ci.iloc[2, 1]

        row = {
            "outcome": label,
            "n_obs": len(Y_clean),
            "n_left": n_left,
            "n_right": n_right,
            "bw_mse": bw,
            "coef_conv": coef,
            "se_conv": se,
            "pval_conv": pval,
            "ci_lower_conv": ci_lower,
            "ci_upper_conv": ci_upper,
            "coef_bc": coef_bc,
            "se_bc": se_bc,
            "pval_bc": pval_bc,
            "coef_rb": coef_rb,
            "se_rb": se_rb,
            "pval_rb": pval_rb,
            "ci_lower_rb": ci_rb_lower,
            "ci_upper_rb": ci_rb_upper,
        }

        stars = ""
        if pval_rb < 0.01:
            stars = "***"
        elif pval_rb < 0.05:
            stars = "**"
        elif pval_rb < 0.10:
            stars = "*"

        print(f"  {label}:")
        print(f"    Robust: coef={coef_rb:.4f}, se={se_rb:.4f}, p={pval_rb:.4f}{stars}")
        print(f"    BW={bw:.4f}, N_left={n_left}, N_right={n_right}")

        return row

    except Exception as e:
        print(f"  {label}: rdrobust failed — {e}")
        return None


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)

    # Load panels
    minor = pd.read_csv(os.path.join(FINAL_DIR, "panel_minor.csv"))
    minor_no_ind = pd.read_csv(os.path.join(FINAL_DIR, "panel_minor_no_ind.csv"))

    results = []

    # =========================================================================
    # Panel A: Minor parties excluding independents (main specification)
    # =========================================================================
    print("=" * 70)
    print("PANEL A: Minor parties excluding independents")
    print("=" * 70)

    df = minor_no_ind.copy()
    X = df["running_var"]

    # Outcome 1: Next-election vote share (conditional on contesting)
    Y1 = df.loc[df["contests_next"] == 1, "next_vote_share"]
    X1 = df.loc[df["contests_next"] == 1, "running_var"]
    r = run_rd(Y1, X1, "Next vote share (conditional)")
    if r:
        r["panel"] = "A: Minor excl. indep."
        results.append(r)

    # Outcome 2: Contests next election
    Y2 = df["contests_next"].astype(float)
    r = run_rd(Y2, X, "Contests next election")
    if r:
        r["panel"] = "A: Minor excl. indep."
        results.append(r)

    # Outcome 3: Next-election vote share (unconditional, 0 if not contesting)
    Y3 = df["next_vote_share_uncond"]
    r = run_rd(Y3, X, "Next vote share (unconditional)")
    if r:
        r["panel"] = "A: Minor excl. indep."
        results.append(r)

    # Outcome 4: Change in vote share (conditional)
    Y4 = df.loc[df["contests_next"] == 1, "vote_share_change"]
    X4 = df.loc[df["contests_next"] == 1, "running_var"]
    r = run_rd(Y4, X4, "Vote share change (conditional)")
    if r:
        r["panel"] = "A: Minor excl. indep."
        results.append(r)

    # =========================================================================
    # Panel B: All minor parties including independents
    # =========================================================================
    print("\n" + "=" * 70)
    print("PANEL B: All minor parties including independents")
    print("=" * 70)

    df = minor.copy()
    X = df["running_var"]

    Y1 = df.loc[df["contests_next"] == 1, "next_vote_share"]
    X1 = df.loc[df["contests_next"] == 1, "running_var"]
    r = run_rd(Y1, X1, "Next vote share (conditional)")
    if r:
        r["panel"] = "B: All minor"
        results.append(r)

    Y2 = df["contests_next"].astype(float)
    r = run_rd(Y2, X, "Contests next election")
    if r:
        r["panel"] = "B: All minor"
        results.append(r)

    Y3 = df["next_vote_share_uncond"]
    r = run_rd(Y3, X, "Next vote share (unconditional)")
    if r:
        r["panel"] = "B: All minor"
        results.append(r)

    Y4 = df.loc[df["contests_next"] == 1, "vote_share_change"]
    X4 = df.loc[df["contests_next"] == 1, "running_var"]
    r = run_rd(Y4, X4, "Vote share change (conditional)")
    if r:
        r["panel"] = "B: All minor"
        results.append(r)

    # =========================================================================
    # Save results
    # =========================================================================
    results_df = pd.DataFrame(results)
    outpath = os.path.join(TABLE_DIR, "table4_rd_main_results.csv")
    results_df.to_csv(outpath, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary_cols = ["panel", "outcome", "coef_rb", "se_rb", "pval_rb", "bw_mse", "n_left", "n_right"]
    print(results_df[summary_cols].to_string(index=False))
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
