"""
06_validity_tests.py
McCrary density test, covariate balance tests, and placebo cutoff tests.
"""

import os
import pandas as pd
import numpy as np
from rdrobust import rdrobust
from rddensity import rddensity

FINAL_DIR = os.path.join(os.path.dirname(__file__), "data", "final")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "output", "tables")


def run_rd_quiet(Y, X, c=0):
    """Run rdrobust, return (coef_rb, se_rb, pval_rb, bw, n_left, n_right) or None."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = ~(np.isnan(Y) | np.isnan(X))
    Y, X = Y[mask], X[mask]
    if len(Y) < 50:
        return None
    try:
        result = rdrobust(Y, X, c=c)
        return {
            "coef": result.coef.iloc[2, 0],
            "se": result.se.iloc[2, 0],
            "pval": result.pv.iloc[2, 0],
            "bw": result.bws.iloc[0, 0],
            "n_left": int(result.N_h[0]),
            "n_right": int(result.N_h[1]),
        }
    except Exception:
        return None


def mccrary_test(df):
    """Run McCrary (rddensity) manipulation test."""
    print("=" * 70)
    print("TEST 1: McCrary Density Test (rddensity)")
    print("=" * 70)

    X = df["running_var"].to_numpy().astype(float)
    X = X[~np.isnan(X)]

    try:
        result = rddensity(X, c=0)
        print(f"  Test statistic: {result.test[0]:.4f}")
        print(f"  P-value: {result.test[1]:.4f}")
        if result.test[1] < 0.05:
            print("  ** Significant at 5% — possible manipulation concern **")
        else:
            print("  Not significant — no evidence of manipulation")
        return {"test_stat": result.test[0], "pval": result.test[1]}
    except Exception as e:
        print(f"  rddensity failed: {e}")
        # Fallback: simple bin-based density test
        print("  Fallback: comparing bin counts around threshold...")
        bins_below = np.sum((X >= -0.005) & (X < 0))
        bins_above = np.sum((X >= 0) & (X < 0.005))
        print(f"  Obs in [-0.5pp, 0): {bins_below}")
        print(f"  Obs in [0, +0.5pp): {bins_above}")
        return {"test_stat": np.nan, "pval": np.nan,
                "n_below_halfpp": bins_below, "n_above_halfpp": bins_above}


def covariate_balance(df):
    """Test for discontinuities in pre-determined covariates at the threshold."""
    print("\n" + "=" * 70)
    print("TEST 2: Covariate Balance (Placebo Outcomes)")
    print("=" * 70)

    results = []
    X = df["running_var"].to_numpy().astype(float)

    # 1. Lagged vote share (from t-1)
    Y_lag = df["lagged_vote_share"].to_numpy().astype(float)
    r = run_rd_quiet(Y_lag, X)
    label = "Lagged vote share (t-1)"
    if r:
        print(f"  {label}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}")
        r["covariate"] = label
        results.append(r)
    else:
        print(f"  {label}: Could not estimate")

    # 2. Number of candidates in division
    Y_ncand = df["n_candidates"].to_numpy().astype(float)
    r = run_rd_quiet(Y_ncand, X)
    label = "N candidates in division"
    if r:
        print(f"  {label}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}")
        r["covariate"] = label
        results.append(r)
    else:
        print(f"  {label}: Could not estimate")

    # 3. Total formal votes in division (division size)
    Y_total = df["total_formal_votes"].to_numpy().astype(float)
    r = run_rd_quiet(Y_total, X)
    label = "Total formal votes (division size)"
    if r:
        print(f"  {label}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}")
        r["covariate"] = label
        results.append(r)
    else:
        print(f"  {label}: Could not estimate")

    return results


def placebo_cutoffs(df):
    """Run RD at fake cutoffs to check for spurious effects."""
    print("\n" + "=" * 70)
    print("TEST 3: Placebo Cutoffs")
    print("=" * 70)

    # Primary outcome: next vote share (conditional on contesting)
    cond = df["contests_next"] == 1
    Y = df.loc[cond, "next_vote_share"].to_numpy().astype(float)
    vote_share = df.loc[cond, "vote_share"].to_numpy().astype(float)

    results = []
    cutoffs = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]

    for c in cutoffs:
        X = vote_share - c
        r = run_rd_quiet(Y, X, c=0)
        label = f"Cutoff at {c*100:.0f}%"
        if r:
            stars = ""
            if r["pval"] < 0.01:
                stars = "***"
            elif r["pval"] < 0.05:
                stars = "**"
            elif r["pval"] < 0.10:
                stars = "*"
            print(f"  {label}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}{stars}")
            r["cutoff"] = c
            results.append(r)
        else:
            print(f"  {label}: Could not estimate")

    print("\n  (Only the 4% cutoff should show an effect if our design is valid)")
    return results


def donut_test(df):
    """Donut hole test: drop observations very close to threshold and re-estimate."""
    print("\n" + "=" * 70)
    print("TEST 4: Donut Hole Test")
    print("=" * 70)

    results = []
    donut_sizes = [0.001, 0.002, 0.005]  # Drop within ±0.1pp, ±0.2pp, ±0.5pp

    for donut in donut_sizes:
        df_donut = df[np.abs(df["running_var"]) >= donut].copy()
        cond = df_donut["contests_next"] == 1
        Y = df_donut.loc[cond, "next_vote_share"].to_numpy().astype(float)
        X = df_donut.loc[cond, "running_var"].to_numpy().astype(float)
        r = run_rd_quiet(Y, X)
        label = f"Donut ±{donut*100:.1f}pp"
        if r:
            print(f"  {label}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}, n={r['n_left']+r['n_right']}")
            r["donut_size"] = donut
            results.append(r)
        else:
            print(f"  {label}: Could not estimate")

    return results


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(FINAL_DIR, "panel_minor_no_ind.csv"))

    # 1. McCrary density test
    mccrary = mccrary_test(df)

    # 2. Covariate balance
    cov_results = covariate_balance(df)

    # 3. Placebo cutoffs
    placebo_results = placebo_cutoffs(df)

    # 4. Donut hole
    donut_results = donut_test(df)

    # Save all results
    all_results = {
        "mccrary": mccrary,
        "covariate_balance": cov_results,
        "placebo_cutoffs": placebo_results,
        "donut_hole": donut_results,
    }

    # Save placebo cutoffs as table
    if placebo_results:
        pd.DataFrame(placebo_results).to_csv(
            os.path.join(TABLE_DIR, "table5_placebo_cutoffs.csv"), index=False
        )

    # Save covariate balance as table
    if cov_results:
        pd.DataFrame(cov_results).to_csv(
            os.path.join(TABLE_DIR, "table6_covariate_balance.csv"), index=False
        )

    # Save donut results
    if donut_results:
        pd.DataFrame(donut_results).to_csv(
            os.path.join(TABLE_DIR, "table7_donut_hole.csv"), index=False
        )

    print(f"\nSaved validity test tables to {TABLE_DIR}/")


if __name__ == "__main__":
    main()
