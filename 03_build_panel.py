"""
03_build_panel.py
Build the party × division × election panel, linking consecutive elections
for the regression discontinuity analysis.
"""

import os
import pandas as pd
import numpy as np

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "data", "interim")
FINAL_DIR = os.path.join(os.path.dirname(__file__), "data", "final")

# AEC public funding rate per eligible vote (approximate, CPI-indexed)
FUNDING_RATES = {
    2004: 1.94,   # ~$1.94 per vote
    2007: 2.10,
    2010: 2.31,
    2013: 2.49,
    2016: 2.63,
    2019: 2.76,
    2022: 2.91,
}

# Consecutive election pairs
ELECTION_PAIRS = [
    (2004, 2007),
    (2007, 2010),
    (2010, 2013),
    (2013, 2016),
    (2016, 2019),
    (2019, 2022),
]

# Divisions that changed names between elections (old -> new)
# These are known redistributions where we can link them
DIVISION_RENAMES = {
    # 2004->2007
    # Gwydir abolished, Flynn created — no direct successor, so we drop these
    # 2007->2010
    # Kalgoorlie->Durack (broadly), Lowe->McMahon, Prospect->Wright — approximate
    # These are not clean 1:1 renames; safer to drop
    # 2013->2016
    # Charlton->Shortland (approx), Fraser->Fenner (ACT), Throsby->Whitlam
    ("Fraser", 2013, "Fenner", 2016),   # ACT redistribution
    # 2016->2019
    ("Batman", 2016, "Cooper", 2019),
    ("Denison", 2016, "Clark", 2019),
    ("Melbourne Ports", 2016, "Macnamara", 2019),
    ("McMillan", 2016, "Monash", 2019),
    ("Murray", 2016, "Nicholls", 2019),
    ("Port Adelaide", 2016, "Spence", 2019),
    ("Wakefield", 2016, "Fraser", 2019),  # Note: Fraser reused
    # 2019->2022
    ("Stirling", 2019, "Hawke", 2022),
}


def build_panel():
    df = pd.read_csv(os.path.join(INTERIM_DIR, "all_first_prefs.csv"))

    # Build a rename mapping for linking: (division, year) -> canonical_division
    # We'll normalize division names so old names map to new names
    rename_map = {}
    for old_div, old_yr, new_div, new_yr in DIVISION_RENAMES:
        rename_map[(old_div, old_yr)] = new_div

    # Create a canonical division name for linking
    df["division_link"] = df.apply(
        lambda r: rename_map.get((r["division"], r["year"]), r["division"]),
        axis=1,
    )

    # RD variables
    df["above_threshold"] = (df["vote_share"] >= 0.04).astype(int)
    df["running_var"] = df["vote_share"] - 0.04

    # Funding amount (imputed)
    df["funding_rate"] = df["year"].map(FUNDING_RATES)
    df["funding_amount"] = df["above_threshold"] * df["votes"] * df["funding_rate"]

    # Number of candidates in division at election t
    n_cands = df.groupby(["year", "division"]).size().reset_index(name="n_candidates")
    df = df.merge(n_cands, on=["year", "division"])

    # Now link consecutive elections
    # For each election pair (t, t+1), merge party × division
    panels = []
    for yr_t, yr_t1 in ELECTION_PAIRS:
        df_t = df[df["year"] == yr_t].copy()
        df_t1 = df[df["year"] == yr_t1].copy()

        # Use party_std and division_link for matching
        # At time t+1, use the actual division name as the link target
        # (since we renamed t's division to match t+1's name)
        df_t1_agg = df_t1.groupby(["party_std", "division"]).agg(
            next_vote_share=("vote_share", "first"),
            next_votes=("votes", "first"),
            next_total_formal_votes=("total_formal_votes", "first"),
        ).reset_index()

        # Merge: t's division_link should match t+1's actual division name
        merged = df_t.merge(
            df_t1_agg,
            left_on=["party_std", "division_link"],
            right_on=["party_std", "division"],
            how="left",
            suffixes=("", "_next"),
        )

        # Contests next election indicator
        merged["contests_next"] = merged["next_vote_share"].notna().astype(int)

        # Fill non-contesting with 0 for unconditional analysis
        merged["next_vote_share_uncond"] = merged["next_vote_share"].fillna(0)

        # Change in vote share (conditional on contesting)
        merged["vote_share_change"] = merged["next_vote_share"] - merged["vote_share"]

        # Election pair identifier
        merged["election_pair"] = f"{yr_t}-{yr_t1}"
        merged["next_year"] = yr_t1

        panels.append(merged)

    panel = pd.concat(panels, ignore_index=True)

    # Clean up column names - drop the duplicate division column from merge
    if "division_next" in panel.columns:
        panel = panel.drop(columns=["division_next"])

    return panel


def add_lagged_vote_share(panel, df):
    """Add lagged vote share (from t-1) for covariate balance test."""
    # For each observation at time t, find the same party×division at t-1
    lag_pairs = {
        2007: 2004,
        2010: 2007,
        2013: 2010,
        2016: 2013,
        2019: 2016,
        2022: 2019,
    }

    # Build a lookup from the full data
    df["division_link"] = df.apply(
        lambda r: r["division"], axis=1  # simplified for lag
    )

    lag_data = df[["year", "party_std", "division", "vote_share"]].copy()
    lag_data = lag_data.rename(columns={"vote_share": "lagged_vote_share", "year": "lag_year"})

    # For each row in panel, look up the vote share at t-1
    panel["lag_year"] = panel["year"].map(lag_pairs)
    panel = panel.merge(
        lag_data,
        left_on=["lag_year", "party_std", "division"],
        right_on=["lag_year", "party_std", "division"],
        how="left",
    )
    panel = panel.drop(columns=["lag_year"])

    return panel


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)

    print("Building panel...")
    panel = build_panel()

    # Add lagged vote share
    df = pd.read_csv(os.path.join(INTERIM_DIR, "all_first_prefs.csv"))
    panel = add_lagged_vote_share(panel, df)

    # Summary
    print(f"\nFull panel: {len(panel)} observations")
    print(f"Election pairs: {panel['election_pair'].value_counts().sort_index().to_string()}")
    print(f"\nBy major/minor:")
    print(f"  Major parties: {panel['is_major'].sum()}")
    print(f"  Minor parties: {(~panel['is_major']).sum()}")
    print(f"  Independents: {panel['is_independent'].sum()}")

    # Minor party panel (main analysis sample)
    minor = panel[~panel["is_major"]].copy()
    print(f"\nMinor party panel: {len(minor)} observations")
    print(f"  Near threshold (1-7%): {((minor['vote_share'] >= 0.01) & (minor['vote_share'] <= 0.07)).sum()}")
    print(f"  Contests next: {minor['contests_next'].sum()} ({minor['contests_next'].mean():.1%})")
    print(f"  Above 4%: {minor['above_threshold'].sum()} ({minor['above_threshold'].mean():.1%})")

    # Minor party excluding independents
    minor_no_ind = minor[~minor["is_independent"]].copy()
    print(f"\nMinor party excl. independents: {len(minor_no_ind)} observations")
    print(f"  Near threshold (1-7%): {((minor_no_ind['vote_share'] >= 0.01) & (minor_no_ind['vote_share'] <= 0.07)).sum()}")
    print(f"  Contests next: {minor_no_ind['contests_next'].sum()} ({minor_no_ind['contests_next'].mean():.1%})")

    # Save
    panel.to_csv(os.path.join(FINAL_DIR, "panel_full.csv"), index=False)
    minor.to_csv(os.path.join(FINAL_DIR, "panel_minor.csv"), index=False)
    minor_no_ind.to_csv(os.path.join(FINAL_DIR, "panel_minor_no_ind.csv"), index=False)

    print(f"\nSaved panel datasets to {FINAL_DIR}/")


if __name__ == "__main__":
    main()
