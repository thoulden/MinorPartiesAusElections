"""
02_clean_data.py
Parse raw AEC DOP CSVs, extract first preference counts,
harmonize party names, and produce a clean candidate-level dataset.
"""

import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
INTERIM_DIR = os.path.join(os.path.dirname(__file__), "data", "interim")

YEARS = [2004, 2007, 2010, 2013, 2016, 2019, 2022]

# Major parties to flag (will be excluded from main RD analysis)
MAJOR_PARTY_CODES = {
    "ALP",   # Australian Labor Party
    "LP",    # Liberal Party
    "LNP",   # Liberal National Party (QLD)
    "NP",    # National Party
    "NCP",   # National Country Party (older name)
    "CLP",   # Country Liberal Party (NT)
    "GRN",   # The Greens
}

# Party name harmonization: map raw AEC abbreviation -> standardized abbreviation
# This groups parties that are the same entity across elections
PARTY_HARMONIZE = {
    # One Nation
    "HAN": "ON",       # 2004 used HAN for Hanson / One Nation
    "ON": "ON",
    # Palmer / United Australia Party
    "PUP": "UAPP",     # Palmer United Party (2013-2016)
    "UAPP": "UAPP",    # United Australia Party (2019-2022)
    # Family First
    "FFP": "FFP",
    # Liberal Democrats
    "LDP": "LDP",
    # Australian Democrats
    "DEM": "DEM",
    # Democratic Labor Party
    "DLP": "DLP",
    # Katter's Australian Party
    "KAP": "KAP",
    # Shooters, Fishers and Farmers
    "SFP": "SFP",
    "SPP": "SPP",      # Shooters and Fishers Party / Shooters Fishers Farmers
    # Nick Xenophon Team / Centre Alliance
    "XEN": "XEN",
    "CA": "XEN",       # Centre Alliance (was NXT)
    # Christian Democratic Party (Fred Nile)
    "CDP": "CDP",
    # Citizens Electoral Council
    "CEC": "CEC",
    # Socialist Alliance
    "SAL": "SAL",
    # Animal Justice Party
    "AJP": "AJP",
    # Independents
    "IND": "IND",
}


def extract_first_prefs(filepath, year):
    """Extract first preference vote counts from a DOP CSV file."""
    df = pd.read_csv(filepath, skiprows=1)

    # First preferences: CountNumber == 0, CalculationType == 'Preference Count'
    fp = df[
        (df["CountNumber"] == 0)
        & (df["CalculationType"] == "Preference Count")
    ].copy()

    # Compute total formal votes per division
    div_totals = fp.groupby("DivisionNm")["CalculationValue"].sum().reset_index()
    div_totals.columns = ["DivisionNm", "total_formal_votes"]

    fp = fp.merge(div_totals, on="DivisionNm")

    # Rename columns for consistency
    fp = fp.rename(columns={
        "StateAb": "state",
        "DivisionID": "division_id",
        "DivisionNm": "division",
        "CandidateID": "candidate_id",
        "Surname": "surname",
        "GivenNm": "given_name",
        "PartyAb": "party_ab",
        "PartyNm": "party_name",
        "CalculationValue": "votes",
        "BallotPosition": "ballot_position",
    })

    fp["year"] = year
    fp["vote_share"] = fp["votes"] / fp["total_formal_votes"]

    # Keep relevant columns
    cols = [
        "year", "state", "division_id", "division",
        "candidate_id", "surname", "given_name", "ballot_position",
        "party_ab", "party_name",
        "votes", "total_formal_votes", "vote_share",
    ]
    return fp[cols].copy()


def harmonize_parties(df):
    """Standardize party abbreviations across elections."""
    # Apply harmonization mapping where available
    df["party_std"] = df["party_ab"].map(PARTY_HARMONIZE).fillna(df["party_ab"])

    # Flag major parties
    df["is_major"] = df["party_ab"].isin(MAJOR_PARTY_CODES)

    # Flag independents
    df["is_independent"] = df["party_ab"] == "IND"

    return df


def main():
    os.makedirs(INTERIM_DIR, exist_ok=True)

    all_years = []
    for year in YEARS:
        filepath = os.path.join(RAW_DIR, f"HouseDopByDivision{year}.csv")
        if not os.path.exists(filepath):
            print(f"Skipping {year}: file not found")
            continue

        print(f"Processing {year}...")
        fp = extract_first_prefs(filepath, year)
        all_years.append(fp)
        print(f"  {len(fp)} candidates across {fp['division'].nunique()} divisions")

    df = pd.concat(all_years, ignore_index=True)
    df = harmonize_parties(df)

    # Summary
    print(f"\nTotal records: {len(df)}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Major party records: {df['is_major'].sum()}")
    print(f"Minor party records: {(~df['is_major']).sum()}")
    print(f"Independent records: {df['is_independent'].sum()}")

    # Save
    outpath = os.path.join(INTERIM_DIR, "all_first_prefs.csv")
    df.to_csv(outpath, index=False)
    print(f"\nSaved to {outpath}")

    # Also print party frequency table
    print("\nParty frequency (standardized):")
    party_counts = df.groupby("party_std").size().sort_values(ascending=False)
    print(party_counts.to_string())


if __name__ == "__main__":
    main()
