"""
04_summary_stats.py
Descriptive statistics and histograms of vote shares around the 4% threshold.
"""

import os
import pandas as pd
import numpy as np

FINAL_DIR = os.path.join(os.path.dirname(__file__), "data", "final")
TABLE_DIR = os.path.join(os.path.dirname(__file__), "output", "tables")


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)

    panel = pd.read_csv(os.path.join(FINAL_DIR, "panel_minor.csv"))
    panel_no_ind = pd.read_csv(os.path.join(FINAL_DIR, "panel_minor_no_ind.csv"))

    # ----- Table 1: Summary statistics -----
    print("=" * 70)
    print("TABLE 1: Summary Statistics — Minor Party Candidates (excl. independents)")
    print("=" * 70)

    df = panel_no_ind.copy()

    stats = {
        "N observations": len(df),
        "N election pairs": df["election_pair"].nunique(),
        "N unique divisions": df["division"].nunique(),
        "N unique parties (std)": df["party_std"].nunique(),
        "": "",
        "Vote share (t)": "",
        "  Mean": f"{df['vote_share'].mean():.4f}",
        "  Median": f"{df['vote_share'].median():.4f}",
        "  Std Dev": f"{df['vote_share'].std():.4f}",
        "  Min": f"{df['vote_share'].min():.4f}",
        "  Max": f"{df['vote_share'].max():.4f}",
        " ": "",
        "Above 4% threshold": f"{df['above_threshold'].mean():.3f}",
        "Contests next election": f"{df['contests_next'].mean():.3f}",
        "  ": "",
        "Next vote share (conditional)": "",
        "  Mean ": f"{df.loc[df['contests_next']==1, 'next_vote_share'].mean():.4f}",
        "  Median ": f"{df.loc[df['contests_next']==1, 'next_vote_share'].median():.4f}",
        "  Std Dev ": f"{df.loc[df['contests_next']==1, 'next_vote_share'].std():.4f}",
        "   ": "",
        "Vote share change (conditional)": "",
        "  Mean  ": f"{df.loc[df['contests_next']==1, 'vote_share_change'].mean():.4f}",
        "  Median  ": f"{df.loc[df['contests_next']==1, 'vote_share_change'].median():.4f}",
        "    ": "",
        "Funding amount ($, imputed)": "",
        "  Mean (above threshold)": f"{df.loc[df['above_threshold']==1, 'funding_amount'].mean():,.0f}",
        "  Median (above threshold)": f"{df.loc[df['above_threshold']==1, 'funding_amount'].median():,.0f}",
    }

    for k, v in stats.items():
        print(f"  {k}: {v}" if v else f"  {k}")

    # Save as CSV
    stats_df = pd.DataFrame(list(stats.items()), columns=["Statistic", "Value"])
    stats_df.to_csv(os.path.join(TABLE_DIR, "table1_summary_stats.csv"), index=False)

    # ----- Table 2: Observations by election pair -----
    print("\n" + "=" * 70)
    print("TABLE 2: Observations by Election Pair")
    print("=" * 70)

    pair_stats = df.groupby("election_pair").agg(
        n_obs=("vote_share", "size"),
        mean_vote_share=("vote_share", "mean"),
        pct_above_4=("above_threshold", "mean"),
        pct_contests_next=("contests_next", "mean"),
        n_divisions=("division", "nunique"),
        n_parties=("party_std", "nunique"),
    ).round(3)
    print(pair_stats.to_string())
    pair_stats.to_csv(os.path.join(TABLE_DIR, "table2_by_election_pair.csv"))

    # ----- Table 3: Observations by party -----
    print("\n" + "=" * 70)
    print("TABLE 3: Top 15 Parties by Observations")
    print("=" * 70)

    party_stats = df.groupby("party_std").agg(
        n_obs=("vote_share", "size"),
        mean_vote_share=("vote_share", "mean"),
        pct_above_4=("above_threshold", "mean"),
        pct_contests_next=("contests_next", "mean"),
    ).sort_values("n_obs", ascending=False).head(15).round(3)
    print(party_stats.to_string())
    party_stats.to_csv(os.path.join(TABLE_DIR, "table3_by_party.csv"))

    # ----- Distribution around threshold -----
    print("\n" + "=" * 70)
    print("Distribution of vote shares around 4% threshold")
    print("=" * 70)

    bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 1.0]
    labels = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%", "5-6%", "6-7%", "7-8%", "8-10%", "10-15%", "15-20%", "20%+"]
    df["vote_share_bin"] = pd.cut(df["vote_share"], bins=bins, labels=labels, right=False)
    bin_counts = df["vote_share_bin"].value_counts().sort_index()
    print(bin_counts.to_string())

    print(f"\nSaved tables to {TABLE_DIR}/")


if __name__ == "__main__":
    main()
