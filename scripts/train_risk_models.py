"""
scripts.train_risk_models — Train the Risk Engine v1 ML Models

1. Loads the Weekly and Monthly v1 data.
2. Trains a LightGBM Regressor for the 7-day Weekly horizon.
3. Trains a RandomForest Regressor for the 30-day Monthly horizon.
4. Serializes the models to `julia/models/*.pkl` and exports feature schemas.
"""

import json
import logging
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Try importing lightgbm; if not installed, fallback to HistGradientBoostingRegressor
try:
    import lightgbm as lgb
    USE_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_LIGHTGBM = False

logger = logging.getLogger("train_risk_models")

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "julia" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FILE_WEEKLY = DATA_DIR / "v1_weekly_train.csv"
FILE_MONTHLY = DATA_DIR / "v1_monthly_train.csv"

# Global Features expected by models
ML_FEATURES = [
    "hurricane_season_flag", 
    "dry_season_flag", 
    "day_length_minutes", 
    "rainfall_last_7_days", 
    "rainfall_last_30_days", 
    "rainfall_anomaly_percent", 
    "TMAX", 
    "heat_stress_flag", 
    "drought_index", 
    "drought_severity_flag",
    "enso_phase_encoded", 
    "enso_strength", 
    "min_distance_to_PR_last_7d", 
    "storm_vmax", 
    "storm_count_last_30_days"
]

def load_data(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset: {file_path}")
    df = pd.read_csv(file_path)
    # Ensure no NaNs in features
    df[ML_FEATURES] = df[ML_FEATURES].fillna(0)
    return df

def train_weekly_model(df: pd.DataFrame):
    """Train the Weekly ML Model using LightGBM (or HistGradientBoost)."""
    logger.info(f"--- Training Weekly Model (Rows: {len(df)}) ---")
    
    X = df[ML_FEATURES]
    y = df["Y_week"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if USE_LIGHTGBM:
        logger.info("Using LightGBM Regressor")
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    else:
        logger.info("Using sklearn HistGradientBoostingRegressor (LightGBM not installed)")
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42)
        
    model.fit(X_train, y_train)
    
    # Eval
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Weekly Model R2: {r2:.3f}, MSE: {mse:.4f}")
    
    # Save Model
    model_path = MODELS_DIR / "weekly_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved weekly model to {model_path}")
    
    return model

def train_monthly_model(df: pd.DataFrame):
    """Train the Monthly ML Model using Random Forest."""
    logger.info(f"--- Training Monthly Model (Rows: {len(df)}) ---")
    
    X = df[ML_FEATURES]
    y = df["Y_month"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Using RandomForest Regressor")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Eval
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Monthly Model R2: {r2:.3f}, MSE: {mse:.4f}")
    
    # Save Model
    model_path = MODELS_DIR / "monthly_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved monthly model to {model_path}")
    
    # Save schema config for the Engine to know what features to pass
    schema = {
        "features": ML_FEATURES,
        "weekly_model_path": "julia/models/weekly_model.pkl",
        "monthly_model_path": "julia/models/monthly_model.pkl"
    }
    schema_path = MODELS_DIR / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Saved feature schema to {schema_path}")
    
    return model

def run():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    df_week = load_data(FILE_WEEKLY)
    df_month = load_data(FILE_MONTHLY)
    
    train_weekly_model(df_week)
    train_monthly_model(df_month)

    logger.info("Risk Model v1 ML Training Complete!")

if __name__ == "__main__":
    run()
