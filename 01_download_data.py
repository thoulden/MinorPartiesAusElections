"""
01_download_data.py
Download AEC House of Representatives Distribution of Preferences data
from the eechidna GitHub repository (originally sourced from AEC Tally Room).
"""

import os
import requests
import time

BASE_URL = "https://raw.githubusercontent.com/jforbes14/eechidna/master/data-raw/elections/data/aec"

# Election years and their AEC election IDs
ELECTIONS = {
    2004: 12246,
    2007: 13745,
    2010: 15508,
    2013: 17496,
    2016: 20499,
    2019: 24310,
    2022: 27966,
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def download_file(url, dest_path, retries=3):
    """Download a file with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                print(f"  Downloaded: {os.path.basename(dest_path)} ({len(r.content):,} bytes)")
                return True
            else:
                print(f"  HTTP {r.status_code} for {url}")
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return False


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download DOP files for each election year (contains first preferences at CountNumber=0)
    for year, eid in ELECTIONS.items():
        filename = f"HouseDopByDivision{year}.csv"
        url = f"{BASE_URL}/{filename}"
        dest = os.path.join(RAW_DIR, filename)

        if os.path.exists(dest):
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading {year} ({eid}): {filename}")
        success = download_file(url, dest)
        if not success:
            print(f"  FAILED to download {filename}")

    # Also try to download 2025 data directly from AEC (ID: 31496)
    # This may not be on GitHub yet
    url_2025 = f"{BASE_URL}/HouseDopByDivision2025.csv"
    dest_2025 = os.path.join(RAW_DIR, "HouseDopByDivision2025.csv")
    if not os.path.exists(dest_2025):
        print("Attempting 2025 data from GitHub...")
        success = download_file(url_2025, dest_2025)
        if not success:
            print("  2025 data not available on GitHub (may need AEC direct download)")

    # Try AEC direct download for 2025
    aec_url_2025 = "https://results.aec.gov.au/31496/Website/Downloads/HouseDopByDivisionDownload-31496.csv"
    if not os.path.exists(dest_2025):
        print("Attempting 2025 data from AEC...")
        success = download_file(aec_url_2025, dest_2025)
        if not success:
            print("  2025 data not available from AEC either — proceeding without it")

    print("\nDownload complete. Files in data/raw/:")
    for f in sorted(os.listdir(RAW_DIR)):
        size = os.path.getsize(os.path.join(RAW_DIR, f))
        print(f"  {f}: {size:,} bytes")


if __name__ == "__main__":
    main()
