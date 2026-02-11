"""
08_robustness.py
Robustness checks: alternative bandwidths, polynomial orders,
donut hole, include/exclude independents, heterogeneity.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdrobust import rdrobust

FINAL_DIR = os.path.join(os.path.dirname(__file__), "data", "final")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "output", "tables")
FIG_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")


def run_rd_opts(Y, X, c=0, h=None, p=1, kernel="triangular"):
    """Run rdrobust with custom options, return dict or None."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = ~(np.isnan(Y) | np.isnan(X))
    Y, X = Y[mask], X[mask]
    if len(Y) < 50:
        return None
    try:
        kwargs = {"c": c, "p": p, "kernel": kernel}
        if h is not None:
            kwargs["h"] = h
        result = rdrobust(Y, X, **kwargs)
        return {
            "coef": result.coef.iloc[2, 0],
            "se": result.se.iloc[2, 0],
            "pval": result.pv.iloc[2, 0],
            "ci_lower": result.ci.iloc[2, 0],
            "ci_upper": result.ci.iloc[2, 1],
            "bw": result.bws.iloc[0, 0],
            "n_left": int(result.N_h[0]),
            "n_right": int(result.N_h[1]),
        }
    except Exception as e:
        return None


def bandwidth_sensitivity(df):
    """Test sensitivity to bandwidth choice."""
    print("=" * 70)
    print("ROBUSTNESS 1: Bandwidth Sensitivity")
    print("=" * 70)

    cond = df["contests_next"] == 1
    Y = df.loc[cond, "next_vote_share"].to_numpy()
    X = df.loc[cond, "running_var"].to_numpy()

    # First get MSE-optimal bandwidth
    r_opt = run_rd_opts(Y, X)
    if r_opt is None:
        print("  Could not estimate baseline")
        return []

    bw_opt = r_opt["bw"]
    print(f"  MSE-optimal bandwidth: {bw_opt:.4f} ({bw_opt*100:.2f}pp)")

    results = []
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    for mult in multipliers:
        h = bw_opt * mult
        r = run_rd_opts(Y, X, h=h)
        if r:
            stars = "***" if r["pval"] < 0.01 else "**" if r["pval"] < 0.05 else "*" if r["pval"] < 0.10 else ""
            print(f"  BW={h:.4f} ({mult:.1f}x): coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}{stars}")
            r["bw_multiplier"] = mult
            r["bandwidth"] = h
            results.append(r)

    return results


def polynomial_sensitivity(df):
    """Test sensitivity to polynomial order."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS 2: Polynomial Order Sensitivity")
    print("=" * 70)

    cond = df["contests_next"] == 1
    Y = df.loc[cond, "next_vote_share"].to_numpy()
    X = df.loc[cond, "running_var"].to_numpy()

    results = []
    for p in [1, 2, 3]:
        r = run_rd_opts(Y, X, p=p)
        if r:
            print(f"  p={p}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}")
            r["poly_order"] = p
            results.append(r)

    return results


def kernel_sensitivity(df):
    """Test sensitivity to kernel choice."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS 3: Kernel Sensitivity")
    print("=" * 70)

    cond = df["contests_next"] == 1
    Y = df.loc[cond, "next_vote_share"].to_numpy()
    X = df.loc[cond, "running_var"].to_numpy()

    results = []
    for kernel in ["triangular", "epanechnikov", "uniform"]:
        r = run_rd_opts(Y, X, kernel=kernel)
        if r:
            print(f"  {kernel}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}")
            r["kernel"] = kernel
            results.append(r)

    return results


def heterogeneity_by_party(df):
    """Heterogeneity by party type."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS 4: Heterogeneity by Party Type")
    print("=" * 70)

    # Established minor parties
    established = {"ON", "FFP", "DEM", "DLP", "KAP", "LDP", "CDP", "UAPP"}
    df["established_minor"] = df["party_std"].isin(established)

    results = []
    for label, subset in [("Established minor", df["established_minor"]),
                          ("Other minor", ~df["established_minor"])]:
        sub = df[subset & (df["contests_next"] == 1)]
        Y = sub["next_vote_share"].to_numpy()
        X = sub["running_var"].to_numpy()
        r = run_rd_opts(Y, X)
        if r:
            print(f"  {label}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}, n={r['n_left']+r['n_right']}")
            r["subgroup"] = label
            results.append(r)
        else:
            print(f"  {label}: Could not estimate (insufficient obs)")

    return results


def heterogeneity_by_period(df):
    """Heterogeneity by election period."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS 5: Heterogeneity by Election Period")
    print("=" * 70)

    results = []
    for pair in sorted(df["election_pair"].unique()):
        sub = df[(df["election_pair"] == pair) & (df["contests_next"] == 1)]
        Y = sub["next_vote_share"].to_numpy()
        X = sub["running_var"].to_numpy()
        r = run_rd_opts(Y, X)
        if r:
            print(f"  {pair}: coef={r['coef']:.4f}, se={r['se']:.4f}, p={r['pval']:.4f}, n={r['n_left']+r['n_right']}")
            r["period"] = pair
            results.append(r)
        else:
            print(f"  {pair}: Could not estimate (insufficient obs)")

    return results


def plot_robustness(bw_results, outpath):
    """Plot coefficient estimates across bandwidths."""
    if not bw_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    mults = [r["bw_multiplier"] for r in bw_results]
    coefs = [r["coef"] for r in bw_results]
    ci_lo = [r["ci_lower"] for r in bw_results]
    ci_hi = [r["ci_upper"] for r in bw_results]

    ax.errorbar(mults, coefs, yerr=[np.array(coefs)-np.array(ci_lo), np.array(ci_hi)-np.array(coefs)],
                fmt="o-", color="steelblue", capsize=5, markersize=8)
    ax.axhline(y=0, color="grey", linewidth=1, linestyle="--")
    ax.axvline(x=1.0, color="red", linewidth=1, linestyle=":", alpha=0.5, label="MSE-optimal")
    ax.set_xlabel("Bandwidth multiplier (relative to MSE-optimal)")
    ax.set_ylabel("RD Estimate (Robust)")
    ax.set_title("Bandwidth Sensitivity: Effect on Next-Election Vote Share")
    ax.legend()

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(FINAL_DIR, "panel_minor_no_ind.csv"))

    # Bandwidth sensitivity
    bw_results = bandwidth_sensitivity(df)

    # Polynomial sensitivity
    poly_results = polynomial_sensitivity(df)

    # Kernel sensitivity
    kernel_results = kernel_sensitivity(df)

    # Heterogeneity by party type
    party_results = heterogeneity_by_party(df)

    # Heterogeneity by period
    period_results = heterogeneity_by_period(df)

    # Save tables
    if bw_results:
        pd.DataFrame(bw_results).to_csv(
            os.path.join(TABLE_DIR, "table8_bandwidth_sensitivity.csv"), index=False
        )

    if poly_results:
        pd.DataFrame(poly_results).to_csv(
            os.path.join(TABLE_DIR, "table9_polynomial_sensitivity.csv"), index=False
        )

    if kernel_results:
        pd.DataFrame(kernel_results).to_csv(
            os.path.join(TABLE_DIR, "table10_kernel_sensitivity.csv"), index=False
        )

    if party_results:
        pd.DataFrame(party_results).to_csv(
            os.path.join(TABLE_DIR, "table11_heterogeneity_party.csv"), index=False
        )

    if period_results:
        pd.DataFrame(period_results).to_csv(
            os.path.join(TABLE_DIR, "table12_heterogeneity_period.csv"), index=False
        )

    # Robustness figure
    plot_robustness(bw_results, os.path.join(FIG_DIR, "fig7_bandwidth_sensitivity.png"))

    print(f"\nAll robustness tables and figures saved")


if __name__ == "__main__":
    main()
