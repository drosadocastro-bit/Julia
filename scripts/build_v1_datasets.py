"""
scripts.build_v1_datasets — Generate ML-ready datasets for Julia Risk Engine v1

1. Parses daily weather data from JSONs (Arecibo, Adjuntas, San Juan).
2. Computes 7-day and 30-day rolling aggregations (precipitation, max temp).
3. Joins ENSO indices.
4. Generates synthetic labels (Y_week, Y_month) using the v0 ClimateRiskEngine.
5. Exports to `v1_weekly_train.csv` and `v1_monthly_train.csv`.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import timedelta

# We need the v0 engine path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from julia.core.risk_engine import ClimateRiskEngine

logger = logging.getLogger("build_v1_datasets")

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_WEEKLY = DATA_DIR / "v1_weekly_train.csv"
OUTPUT_MONTHLY = DATA_DIR / "v1_monthly_train.csv"

# The three station files we verified
JSON_FILES = [
    DATA_DIR / "data.json",       # Arecibo
    DATA_DIR / "data (1).json",   # Adjuntas / Toa Baja Levittown
    DATA_DIR / "data (2).json"    # San Juan
]

def load_daily_weather_jsons() -> pd.DataFrame:
    """Parse the specific JSON files containing daily PR weather info."""
    dfs = []
    
    for file_path in JSON_FILES:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        metadata = data.get("metadata", {})
        station_id = metadata.get("id", "Unknown")
        station_name = metadata.get("name", "Unknown")
        
        timeseries = data.get("timeseries", {})
        if not timeseries:
            continue
            
        # timeseries format: {"PRCP": {"YYYY-MM-DD": value, ...}, "TMAX": {...}}
        # Create a DF from the dictionary of dictionaries
        df_station = pd.DataFrame(timeseries)
        df_station.index = pd.to_datetime(df_station.index, errors='coerce')
        df_station = df_station.sort_index().dropna(how='all')
        
        df_station["station_id"] = station_id
        df_station["station_name"] = station_name
        
        dfs.append(df_station)
        
    if not dfs:
        return pd.DataFrame()
        
    master_df = pd.concat(dfs).reset_index()
    master_df = master_df.rename(columns={"index": "date"})
    # Convert 'date' correctly if it wasn't
    master_df["date"] = pd.to_datetime(master_df["date"])
    
    # Sort by station and date
    master_df = master_df.sort_values(by=["station_id", "date"]).reset_index(drop=True)
    return master_df

def load_enso_data() -> pd.DataFrame:
    """Load monthly ENSO indices mapping Year-Month to phase."""
    file_path = DATA_DIR / "oni_enso_index_1950_2025.csv"
    if not file_path.exists():
        return pd.DataFrame()
        
    df_enso = pd.read_csv(file_path)
    # The ENSO data uses 'year' and 'season' e.g. DJF, JFM, FMA...
    # We'll create a crude mapping to month for joining based on the middle month of the season
    season_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
    }
    df_enso["month"] = df_enso["season"].map(season_to_month)
    
    # Clean phases
    def encode_phase(p):
        if str(p) == "El Nino": return 1
        if str(p) == "La Nina": return -1
        return 0
        
    df_enso["enso_phase_encoded"] = df_enso["phase"].apply(encode_phase)
    df_enso["enso_strength"] = df_enso["oni"].abs()
    
    return df_enso[["year", "month", "oni", "enso_phase_encoded", "enso_strength"]]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the temporal and rolling features required in Phase 2."""
    
    # Basic Temporal
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["day_of_year"] = df["date"].dt.dayofyear
    
    # Agronomy Flags
    # Hurricane: Jun - Nov (6 - 11)
    df["hurricane_season_flag"] = df["month"].isin([6,7,8,9,10,11]).astype(int)
    # Dry Season: Dec - Apr
    df["dry_season_flag"] = df["month"].isin([12,1,2,3,4]).astype(int)
    
    # Simple day length proxy based on PR latitude (~18N)
    # Annual mean is ~12 hours (720 mins). Summer solstice ~13h15m (795m), Winter ~11h00 (660m)
    # Equation: 720 + 60 * sin((day_of_year - 80) / 365 * 2pi)
    df["day_length_minutes"] = 720.0 + 60.0 * np.sin((df["day_of_year"] - 80) / 365.25 * 2 * np.pi)
    df["delta_from_annual_mean"] = df["day_length_minutes"] - 720.0
    
    # Ensure numeric types and handle NAs before rolling
    for col in ["PRCP", "TMAX"]:
        if col in df.columns:
            # Handle possible string elements safely
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    # Rolling Features per station
    logger.info("Calculating rolling windows (7d and 30d)...")
    df = df.sort_values(by=["station_id", "date"])
    
    grouped = df.groupby("station_id")
    
    # Rainfall
    if "PRCP" in df.columns:
        df["rainfall_last_7_days"] = grouped["PRCP"].transform(lambda x: x.rolling(7, min_periods=1).sum())
        df["rainfall_last_30_days"] = grouped["PRCP"].transform(lambda x: x.rolling(30, min_periods=1).sum())
        # Naive normal is based on the month, let's use 30d trailing average of the WHOLE dataset for this month as normal proxy
        # Since we want a robust pipeline, we'll calculate monthly mean precip overall
        monthly_normals = df.groupby(["station_id", "month"])["PRCP"].mean() * 30 # approx 30 days
        df = df.join(monthly_normals.rename("rainfall_normal_mm"), on=["station_id", "month"])
        df["rainfall_anomaly"] = df["rainfall_last_30_days"] - df["rainfall_normal_mm"]
        df["rainfall_anomaly_percent"] = (df["rainfall_anomaly"] / (df["rainfall_normal_mm"] + 1e-5)) * 100
    else:
        df["rainfall_last_7_days"] = 0.0
        df["rainfall_last_30_days"] = 0.0
        df["rainfall_anomaly"] = 0.0
        df["rainfall_anomaly_percent"] = 0.0
        
    # Heat/Drought
    if "TMAX" in df.columns:
        df["heat_stress_flag"] = (df["TMAX"] > 90).astype(int) # Over 90F
    else:
        df["TMAX"] = 85.0
        df["heat_stress_flag"] = 0
        
    # Proxy Drought Index (Combining long dry spell with heat)
    # E.g. If rainfall_last_30_days is low and heat is high
    # Standard normal mapping: 0 = ok, -3 = severe drought
    df["drought_index"] = np.where(df["rainfall_anomaly_percent"] < -30, -1, 0)
    df["drought_index"] = np.where(df["rainfall_anomaly_percent"] < -60, -2, df["drought_index"])
    df["drought_severity_flag"] = (df["drought_index"] <= -2).astype(int)

    # Note: We won't implement active HURDAT2 logic right now, as joining geospatial storm tracks 
    # to daily point data is a complex Phase 3 logic. We will mock the storm features based
    # on hurricane season presence for now, but will utilize the v0 engine's flexibility.
    # We will randomly spike storm proximity during hurricane season for training entropy.
    np.random.seed(42)
    mask_storm = (df["hurricane_season_flag"] == 1) & (np.random.random(len(df)) < 0.02) # 2% chance of storm passing 
    df["min_distance_to_PR_last_7d"] = np.where(mask_storm, np.random.uniform(10, 300, len(df)), 9999)
    df["storm_vmax"] = np.where(mask_storm, np.random.uniform(50, 160, len(df)), 0)
    df["storm_count_last_30_days"] = np.where(mask_storm, 1, 0)
    df["days_since_last_storm"] = np.where(mask_storm, np.random.uniform(0, 7, len(df)), 7)
    
    return df

def generate_labels(df: pd.DataFrame, engine: ClimateRiskEngine) -> pd.DataFrame:
    """Use the deterministic engine to label the historical dataset horizons."""
    
    logger.info("Computing daily deterministic v0 ground truths...")
    
    def evaluate_row(row):
        res = engine.evaluate(
            storm_dist_km=row.get("min_distance_to_PR_last_7d", 9999),
            storm_vmax=row.get("storm_vmax", 0),
            storm_count=row.get("storm_count_last_30_days", 0),
            storm_days_since=row.get("days_since_last_storm", 7),
            rain_anomaly_pct=row.get("rainfall_anomaly_percent", 0),
            drought_idx=row.get("drought_index", 0),
            day_length_min=row.get("day_length_minutes", 720),
            annual_mean_min=720,
            enso_phase=row.get("enso_phase_encoded", 0)
        )
        return res["composite"]["final_risk"]
    
    # Calculate daily absolute risk
    df["daily_risk_v0"] = df.apply(evaluate_row, axis=1)
    
    # Group by station to calculate future target headers
    logger.info("Generating Y_week (max 7-day future) and Y_month (mean 30-day future)...")
    grouped = df.groupby("station_id")
    
    # Reverse rolling to get future windows
    df = df.sort_values(by=["station_id", "date"])
    
    df["Y_week"] = grouped["daily_risk_v0"].transform(
        lambda x: x.shift(-7).rolling(7, min_periods=1).max()
    )
    df["Y_month"] = grouped["daily_risk_v0"].transform(
        lambda x: x.shift(-30).rolling(30, min_periods=1).mean()
    )
    
    # Drop rows at the terminal edge which lack full future labels
    df = df.dropna(subset=["Y_week", "Y_month"])
    
    return df

def create_weekly_monthly_datasets(df: pd.DataFrame):
    """Aggregate daily dataframe into weekly and monthly ML datasets."""
    
    # Keep useful ML features
    features = [
        "station_id", "date", "year", "month", "week_of_year", "day_of_year", 
        "hurricane_season_flag", "dry_season_flag", "day_length_minutes", 
        "rainfall_last_7_days", "rainfall_last_30_days", "rainfall_anomaly_percent", 
        "TMAX", "heat_stress_flag", "drought_index", "drought_severity_flag",
        "enso_phase_encoded", "enso_strength", 
        "min_distance_to_PR_last_7d", "storm_vmax", "storm_count_last_30_days",
        "Y_week", "Y_month"
    ]
    
    df_clean = df[[c for c in features if c in df.columns]].copy()
    
    # 1. Weekly Dataset (Sample 1 row per week per station)
    df_weekly = df_clean.groupby(["station_id", "year", "week_of_year"]).last().reset_index()
    # Weekly model targets Y_week
    df_weekly = df_weekly.drop(columns=["Y_month"])
    
    # 2. Monthly Dataset (Sample 1 row per month per station)
    df_monthly = df_clean.groupby(["station_id", "year", "month"]).last().reset_index()
    # Monthly model targets Y_month, but we might want to aggegate the preceding month's features
    # For simplicity of this v1 baseline, we take the terminal day of the month's trailing state
    df_monthly = df_monthly.drop(columns=["Y_week"])
    
    # Final cleanup (drop NaN label columns just in case)
    df_weekly = df_weekly.dropna(subset=["Y_week"])
    df_monthly = df_monthly.dropna(subset=["Y_month"])
    
    df_weekly.to_csv(OUTPUT_WEEKLY, index=False)
    df_monthly.to_csv(OUTPUT_MONTHLY, index=False)
    
    logger.info(f"Exported Weekly DB: {OUTPUT_WEEKLY} (Rows: {len(df_weekly)})")
    logger.info(f"Exported Monthly DB: {OUTPUT_MONTHLY} (Rows: {len(df_monthly)})")

def run():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    logger.info("loading daily PR weather JSONs...")
    df = load_daily_weather_jsons()
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
        
    logger.info("Building features (rolling 7d/30d)...")
    df = build_features(df)
    
    logger.info("Loading ENSO data...")
    df_enso = load_enso_data()
    if not df_enso.empty:
        df = df.merge(df_enso, on=["year", "month"], how="left")
        # Fill missing ENSO with neutral
        df["enso_phase_encoded"] = df["enso_phase_encoded"].fillna(0)
        df["enso_strength"] = df["enso_strength"].fillna(0)
    
    engine = ClimateRiskEngine()
    df = generate_labels(df, engine)
    
    create_weekly_monthly_datasets(df)
    logger.info("Dataset Pipeline v1 Complete!")

if __name__ == "__main__":
    run()
