"""
scripts.stress_test_engine — Adversarial Simulation Harness for Julia Risk v1

Injects structured, extreme 45-day scenarios to evaluate the ML engine's stability, 
override temperament, and divergence from the deterministic ground-truth.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from julia.core.risk_engine import ClimateRiskEngine

logger = logging.getLogger("stress_test")

LOGS_DIR = Path(__file__).parent.parent / "julia" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global variables to simulate rolling state in the harness
class EngineState:
    def __init__(self):
        self.rainfall_history = []
        self.storm_history = []  # List of tuples (days_ago, distance, vmax)
        
    def add_day(self, rain_mm: float, storm_dist: float, storm_vmax: float):
        self.rainfall_history.append(rain_mm)
        # Age the storms
        new_storms = []
        for d, dist, v in self.storm_history:
            if d + 1 <= 30:
                new_storms.append((d + 1, dist, v))
                
        # If there's a new storm today (distance < 999), track it
        if storm_dist < 999:
            new_storms.append((0, storm_dist, storm_vmax))
            
        self.storm_history = new_storms
        
    def get_features(self):
        # Rolling 7d rainfall
        rain_7d = sum(self.rainfall_history[-7:]) if len(self.rainfall_history) >= 7 else sum(self.rainfall_history)
        # Rolling 30d rainfall
        rain_30d = sum(self.rainfall_history[-30:]) if len(self.rainfall_history) >= 30 else sum(self.rainfall_history)
        
        # Calculate derived anomaly (Simple estimation for testing: assume 4mm/day normal)
        normal_7d = 28
        normal_30d = 120
        anomaly_pct = ((rain_30d - normal_30d) / normal_30d) * 100 if normal_30d > 0 else 0
        
        # Storm stats
        storm_count_30d = len(self.storm_history)
        
        # Closest storm in last 7 days
        recent_storms = [s for s in self.storm_history if s[0] <= 7]
        min_dist_7d = 9999
        vmax_dist_7d = 0
        days_since_dist = 7
        if recent_storms:
            closest = min(recent_storms, key=lambda x: x[1])
            days_since_dist, min_dist_7d, vmax_dist_7d = closest
            
        # Max storm in last 30 days
        max_vmax_30d = max([s[2] for s in self.storm_history]) if self.storm_history else 0
            
        return {
            "rainfall_last_7_days": rain_7d,
            "rainfall_last_30_days": rain_30d,
            "rainfall_anomaly_percent": anomaly_pct,
            "storm_count_last_30_days": storm_count_30d,
            "min_distance_to_PR_last_7d": min_dist_7d,
            "storm_vmax": max_vmax_30d,         # Actually we need max intensity for the feature
            "days_since_closest": days_since_dist,
            "vmax_closest": vmax_dist_7d
        }

def generate_scenario_1_extreme_storms():
    """3 storms in 45 days. 200km miss, Direct hit (140mph), Weak trailing."""
    data = []
    start_date = datetime(2026, 9, 1)
    
    for i in range(45):
        # Baseline normal weather
        row = {
            "date": start_date + timedelta(days=i),
            "rain_mm": 5.0,
            "temp_max": 31.0,
            "day_len_min": 720,
            "enso_phase": 0,    # Neutral
            "storm_dist": 9999,
            "storm_vmax": 0
        }
        
        # Day 10: Near miss (200km, 80mph)
        if i == 10:
            row["storm_dist"] = 200
            row["storm_vmax"] = 80
            row["rain_mm"] = 40.0
            
        # Day 25: Direct Hit (10km, 140mph Cat 4)
        if i == 25:
            row["storm_dist"] = 10
            row["storm_vmax"] = 140
            row["rain_mm"] = 150.0
            
        # Day 38: Weak tropical depression (50km, 40mph)
        if i == 38:
            row["storm_dist"] = 50
            row["storm_vmax"] = 40
            row["rain_mm"] = 25.0
            
        data.append(row)
    return pd.DataFrame(data), "Scenario 1: Extreme Storm Season"

def generate_scenario_2_flash_drought():
    """40 days below normal rainfall, peak solar, ENSO neutral."""
    data = []
    start_date = datetime(2026, 6, 1)
    
    for i in range(45):
        # Severe rainfall deficit
        row = {
            "date": start_date + timedelta(days=i),
            "rain_mm": 0.5,     # Barely any rain
            "temp_max": 34.0,   # Very hot
            "day_len_min": 800, # Long summer days
            "enso_phase": 0,    # Neutral
            "storm_dist": 9999,
            "storm_vmax": 0
        }
        data.append(row)
    return pd.DataFrame(data), "Scenario 2: Flash Drought"

def generate_scenario_3_la_nina_wet():
    """High rainfall (+70%), frequent moderate storms, ENSO -1."""
    data = []
    start_date = datetime(2026, 10, 1)
    
    for i in range(45):
        row = {
            "date": start_date + timedelta(days=i),
            "rain_mm": 12.0,    # Consistent heavy rain (~3x normal)
            "temp_max": 29.0,
            "day_len_min": 700,
            "enso_phase": -1,   # La Niña
            "storm_dist": 9999,
            "storm_vmax": 0
        }
        
        # Frequent minor squalls
        if i % 14 == 0:
            row["storm_dist"] = 150
            row["storm_vmax"] = 50
            row["rain_mm"] = 60.0
            
        data.append(row)
    return pd.DataFrame(data), "Scenario 3: La Niña Wet Year"

def generate_scenario_4_false_positives():
    """Fake proximity data (0km) but impossible vmax or sudden spikes."""
    data = []
    start_date = datetime(2026, 8, 1)
    
    for i in range(45):
        row = {
            "date": start_date + timedelta(days=i),
            "rain_mm": 5.0,
            "temp_max": 31.0,
            "day_len_min": 720,
            "enso_phase": 0,
            "storm_dist": 9999,
            "storm_vmax": 0
        }
        
        # Day 15: Glitch - 0km distance but 0 vmax
        if i == 15:
            row["storm_dist"] = 0
            row["storm_vmax"] = 0
            
        # Day 28: Impossible Glitch - 500mph
        if i == 28:
            row["storm_dist"] = 500
            row["storm_vmax"] = 500
            
        data.append(row)
    return pd.DataFrame(data), "Scenario 4: False Positive Glitches"

def run_simulation(df: pd.DataFrame, scenario_name: str, engine: ClimateRiskEngine):
    logger.info(f"--- Running {scenario_name} ---")
    
    state = EngineState()
    results = []
    
    for _, row in df.iterrows():
        # 1. Update rolling state
        state.add_day(row["rain_mm"], row["storm_dist"], row["storm_vmax"])
        feats = state.get_features()
        
        # 2. Prepare ML Dictionary payload
        # Map variables to match ML schema expectations
        ml_payload = {
            "hurricane_season_flag": 1 if row["date"].month in [6,7,8,9,10,11] else 0,
            "dry_season_flag": 1 if row["date"].month in [12,1,2,3,4] else 0,
            "day_length_minutes": row["day_len_min"],
            "rainfall_last_7_days": feats["rainfall_last_7_days"],
            "rainfall_last_30_days": feats["rainfall_last_30_days"],
            "rainfall_anomaly_percent": feats["rainfall_anomaly_percent"],
            "TMAX": row["temp_max"],
            "heat_stress_flag": 1 if row["temp_max"] > 32 else 0,
            "drought_index": -2.0 if feats["rainfall_anomaly_percent"] < -50 else 0.0, # Crude proxy
            "drought_severity_flag": 1 if feats["rainfall_anomaly_percent"] < -60 else 0,
            "enso_phase_encoded": row["enso_phase"],
            "enso_strength": abs(row["enso_phase"]),
            "min_distance_to_PR_last_7d": feats["min_distance_to_PR_last_7d"],
            "storm_vmax": feats["storm_vmax"],
            "storm_count_last_30_days": feats["storm_count_last_30_days"]
        }
        
        # 3. Predict v0
        v0_res = engine.evaluate(
            storm_dist_km=feats["min_distance_to_PR_last_7d"], 
            storm_vmax=feats["vmax_closest"], 
            storm_count=feats["storm_count_last_30_days"], 
            storm_days_since=feats["days_since_closest"],
            rain_anomaly_pct=feats["rainfall_anomaly_percent"], 
            drought_idx=ml_payload["drought_index"], 
            day_length_min=row["day_len_min"], 
            annual_mean_min=750,
            enso_phase=row["enso_phase"]
        )
        v0_score = v0_res["composite"]["final_risk"]
        
        # 4. Predict v1 (ML)
        v1_res = engine.evaluate_v1(ml_payload)
        ml_week = v1_res["horizons"]["weekly_risk_score"]
        ml_month = v1_res["horizons"]["monthly_risk_score"]
        ml_comp = v1_res["composite"]["final_risk"]
        advisory = v1_res["composite"]["final_advisory"]
        
        results.append({
            "day": len(results) + 1,
            "date": row["date"].strftime("%Y-%m-%d"),
            "rain_mm": row["rain_mm"],
            "storm_dist": row["storm_dist"],
            "v0_score": v0_score,
            "ml_week_score": ml_week,
            "ml_month_score": ml_month,
            "ml_comp_score": ml_comp,
            "advisory": advisory,
            "is_override": 1 if advisory == "WEEKLY_CRITICAL_OVERRIDE" else 0
        })
        
    res_df = pd.DataFrame(results)
    
    # Calculate Diagnostics
    overrides = res_df["is_override"].sum()
    max_oscillation = res_df["ml_comp_score"].diff().abs().max()
    mae_divergence = (res_df["ml_comp_score"] - res_df["v0_score"]).abs().mean()
    
    logger.info(f" > Total Overrides: {overrides} (Goal < 5 limits burnout)")
    logger.info(f" > Max 24h Oscillation: {max_oscillation:.3f} (Lower = more stable)")
    logger.info(f" > v0 vs ML Divergence (MAE): {mae_divergence:.3f} (Lower = ML aligned with rules)")
    
    # Check bounds
    if overrides > 5:
        logger.warning(" 🚨 SYSTEM IS TOO ANXIOUS! Too many overrides.")
    if max_oscillation > 0.4:
        logger.warning(" 🚨 HIGH OSCILLATION! Score bounces erratically day-to-day.")
        
    return res_df

def run():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = ClimateRiskEngine()
    
    logger.info("====================================")
    logger.info(" JULIA V1 ADVERSARIAL STRESS TEST")
    logger.info("====================================")
    
    scenarios = [
        generate_scenario_1_extreme_storms(),
        generate_scenario_2_flash_drought(),
        generate_scenario_3_la_nina_wet(),
        generate_scenario_4_false_positives()
    ]
    
    all_results = []
    for payload, name in scenarios:
        res = run_simulation(payload, name, engine)
        res["scenario"] = name
        all_results.append(res)
        
    final_df = pd.concat(all_results, ignore_index=True)
    out_path = LOGS_DIR / "stress_test_results.csv"
    final_df.to_csv(out_path, index=False)
    logger.info(f"Saved detailed diagnostics to {out_path}")

if __name__ == "__main__":
    run()
