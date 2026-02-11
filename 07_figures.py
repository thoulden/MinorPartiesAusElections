"""
07_figures.py
RD plots, density plots, and coefficient plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from rdrobust import rdrobust

FINAL_DIR = os.path.join(os.path.dirname(__file__), "data", "final")
FIG_DIR = os.path.join(os.path.dirname(__file__), "output", "figures")


def plot_histogram(df, outpath):
    """Histogram of running variable (vote share - 4%)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    running = df["running_var"] * 100  # Convert to percentage points
    bins = np.arange(-4, 10.5, 0.5)

    ax.hist(running, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linewidth=1.5, linestyle="--", label="4% threshold")
    ax.set_xlabel("Vote share minus 4% (percentage points)")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Distribution of Minor Party Vote Shares Around the 4% Funding Threshold")
    ax.legend()
    ax.set_xlim(-4, 10)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def rdplot_manual(df, y_col, x_col, title, ylabel, outpath, n_bins=20, subset=None):
    """
    Manual RD plot: bin scatter with local polynomial fits on each side.
    """
    if subset is not None:
        df = df[subset].copy()

    Y = df[y_col].to_numpy().astype(float)
    X = df[x_col].to_numpy().astype(float) * 100  # Convert to pp
    mask = ~(np.isnan(Y) | np.isnan(X))
    Y, X = Y[mask], X[mask]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create bin scatter
    # Left side (below threshold)
    x_left = X[X < 0]
    y_left = Y[X < 0]
    # Right side (above threshold)
    x_right = X[X >= 0]
    y_right = Y[X >= 0]

    def bin_scatter(x, y, n_bins):
        if len(x) == 0:
            return [], []
        bins = np.linspace(x.min(), x.max(), n_bins + 1)
        bin_centers = []
        bin_means = []
        for i in range(len(bins) - 1):
            mask = (x >= bins[i]) & (x < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(y[mask].mean())
        return bin_centers, bin_means

    bc_left, bm_left = bin_scatter(x_left, y_left, n_bins)
    bc_right, bm_right = bin_scatter(x_right, y_right, n_bins)

    ax.scatter(bc_left, bm_left, color="steelblue", s=40, zorder=3)
    ax.scatter(bc_right, bm_right, color="steelblue", s=40, zorder=3)

    # Fit local polynomials (degree 1) on each side
    if len(x_left) > 10:
        coef_l = np.polyfit(x_left, y_left, 1)
        x_fit_l = np.linspace(x_left.min(), -0.01, 100)
        ax.plot(x_fit_l, np.polyval(coef_l, x_fit_l), color="darkblue", linewidth=2)

    if len(x_right) > 10:
        coef_r = np.polyfit(x_right, y_right, 1)
        x_fit_r = np.linspace(0.01, x_right.max(), 100)
        ax.plot(x_fit_r, np.polyval(coef_r, x_fit_r), color="darkblue", linewidth=2)

    ax.axvline(x=0, color="red", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_xlabel("Vote share minus 4% (percentage points)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def plot_density(df, outpath):
    """McCrary-style density plot: fine-binned histogram with local linear
    smoothing on each side of the cutoff, avoiding KDE boundary bias."""
    fig, ax = plt.subplots(figsize=(8, 5))

    running = df["running_var"].to_numpy() * 100  # percentage points
    running = running[(running >= -4) & (running <= 8)]

    # Fine bins for the histogram (used for the scatter points)
    bin_width = 0.25  # 0.25pp bins
    bin_edges = np.arange(-4, 8 + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(running, bins=bin_edges)
    # Convert counts to density (counts / n / bin_width)
    density = counts / (len(running) * bin_width)

    # Plot bin scatter separately for left and right
    left_mask = bin_centers < 0
    right_mask = bin_centers >= 0
    ax.scatter(bin_centers[left_mask], density[left_mask],
               color="steelblue", s=25, zorder=3, alpha=0.8)
    ax.scatter(bin_centers[right_mask], density[right_mask],
               color="steelblue", s=25, zorder=3, alpha=0.8)

    # Local linear (LOWESS-style) smooth on each side
    from statsmodels.nonparametric.smoothers_lowess import lowess

    for mask, side_label in [(left_mask, "left"), (right_mask, "right")]:
        x_pts = bin_centers[mask]
        y_pts = density[mask]
        if len(x_pts) > 4:
            smoothed = lowess(y_pts, x_pts, frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="darkblue",
                    linewidth=2, zorder=4)

    ax.axvline(x=0, color="red", linewidth=1.5, linestyle="--", label="4% threshold")
    ax.set_xlabel("Vote share minus 4% (percentage points)")
    ax.set_ylabel("Density")
    ax.set_title("McCrary Density Plot: Vote Shares Around the 4% Threshold")
    ax.legend()
    ax.set_xlim(-4, 8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def plot_coefficient(results_path, outpath):
    """Coefficient plot from main results."""
    results = pd.read_csv(results_path)
    panel_a = results[results["panel"] == "A: Minor excl. indep."].copy()

    if len(panel_a) == 0:
        print("  No Panel A results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(panel_a))
    coefs = panel_a["coef_rb"].values
    ci_low = panel_a["ci_lower_rb"].values
    ci_high = panel_a["ci_upper_rb"].values
    labels = panel_a["outcome"].values

    ax.errorbar(coefs, y_pos, xerr=[coefs - ci_low, ci_high - coefs],
                fmt="o", color="steelblue", capsize=4, markersize=8)
    ax.axvline(x=0, color="grey", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("RD Estimate (Robust)")
    ax.set_title("Main RD Estimates — Effect of Election Funding on Minor Parties")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(outpath)}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(FINAL_DIR, "panel_minor_no_ind.csv"))

    print("Generating figures...")

    # Figure 1: Histogram of running variable
    plot_histogram(df, os.path.join(FIG_DIR, "fig1_histogram_running_var.png"))

    # Figure 2: Density plot
    plot_density(df, os.path.join(FIG_DIR, "fig2_density_running_var.png"))

    # Figure 3: RD plot — next vote share (conditional)
    cond = df["contests_next"] == 1
    # Restrict to reasonable window for visual clarity
    window = (df["running_var"] >= -0.04) & (df["running_var"] <= 0.08)
    rdplot_manual(
        df[cond & window],
        y_col="next_vote_share",
        x_col="running_var",
        title="RD Plot: Next-Election Vote Share (Conditional on Contesting)",
        ylabel="Next election vote share",
        outpath=os.path.join(FIG_DIR, "fig3_rd_next_vote_share_conditional.png"),
        n_bins=15,
    )

    # Figure 4: RD plot — contests next election
    rdplot_manual(
        df[window],
        y_col="contests_next",
        x_col="running_var",
        title="RD Plot: Probability of Contesting Next Election",
        ylabel="Pr(contests next election)",
        outpath=os.path.join(FIG_DIR, "fig4_rd_contests_next.png"),
        n_bins=15,
    )

    # Figure 5: RD plot — unconditional vote share
    rdplot_manual(
        df[window],
        y_col="next_vote_share_uncond",
        x_col="running_var",
        title="RD Plot: Next-Election Vote Share (Unconditional)",
        ylabel="Next election vote share (0 if not contesting)",
        outpath=os.path.join(FIG_DIR, "fig5_rd_next_vote_share_unconditional.png"),
        n_bins=15,
    )

    # Figure 6: Coefficient plot
    results_path = os.path.join(os.path.dirname(__file__), "output", "tables", "table4_rd_main_results.csv")
    if os.path.exists(results_path):
        plot_coefficient(results_path, os.path.join(FIG_DIR, "fig6_coefficient_plot.png"))

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
