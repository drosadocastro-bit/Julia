import requests
import pandas as pd
from datetime import datetime

BASE = "https://usdmdataservices.unl.edu/api"
AREA = "StateStatistics"
AOI_PR = "72"  # Puerto Rico (2-digit FIPS)
STATISTICS_TYPE = 1  # "traditional format" per docs

# We’ll use a "severity by area percent" endpoint.
# If this specific endpoint name differs in your environment, see notes below.
ENDPOINT = f"{BASE}/{AREA}/GetDroughtSeverityStatisticsByAreaPercent"

def fetch_year(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "aoi": AOI_PR,
        "startdate": start_date,  # M/D/YYYY
        "enddate": end_date,      # M/D/YYYY
        "statisticsType": STATISTICS_TYPE
    }
    # Ask for JSON output
    headers = {"Accept": "application/json"}
    r = requests.get(ENDPOINT, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Common shapes: list[dict] or dict with a key like "Data"
    if isinstance(data, dict):
        # Try common keys
        for k in ["Data", "data", "Results", "results"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    df = pd.DataFrame(data)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Detect date
    if "validstart" in df.columns:
        df["date"] = pd.to_datetime(df["validstart"], errors="coerce").dt.date
    elif "mapdate" in df.columns:
        df["date"] = pd.to_datetime(df["mapdate"], errors="coerce").dt.date
    else:
        raise ValueError("No recognizable date column found.")

    # Ensure drought columns exist
    required = ["d0","d1","d2","d3","d4"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing drought columns: {missing}. Columns found: {list(df.columns)}")

    # Convert to numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute normalized drought score (0–1)
    df["drought_score_norm"] = (
        1*df["d0"] +
        2*df["d1"] +
        3*df["d2"] +
        4*df["d3"] +
        5*df["d4"]
    ) / (5 * 100.0)

    return df[["date","d0","d1","d2","d3","d4","drought_score_norm"]].sort_values("date")

def main():
    # USDM: weekly; your report range: ~6/26/2018 to 3/3/2020
    # We'll pull 2018-01-01..2018-12-31, 2019 full year, 2020 partial
    ranges = [
        ("1/1/2018","12/31/2018"),
        ("1/1/2019","12/31/2019"),
        ("1/1/2020","12/31/2020"),
    ]

    dfs = []
    for s,e in ranges:
        print(f"Fetching {s} → {e} ...")
        df = fetch_year(s,e)
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    tidy = normalize_columns(raw)

    tidy.to_csv("pr_usdm_weekly_percent_area.csv", index=False)
    tidy.to_json("pr_usdm_weekly_percent_area.json", orient="records")

    print("Saved:")
    print(" - pr_usdm_weekly_percent_area.csv")
    print(" - pr_usdm_weekly_percent_area.json")
    print("Rows:", len(tidy))
    print(tidy.head())

if __name__ == "__main__":
    main()