import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import os

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from julia.core.database import JuliaDatabase
from julia.core.config import JuliaConfig
from julia.agent import JuliaAgent

logger = logging.getLogger("deep_dive")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_scenario_row(day: int):
    row = {
        "date": datetime(2026, 1, 1) + timedelta(days=day),
        "rain_mm": 5.0,
        "temp_max": 28.0,
        "storm_dist": 9999,
        "soil_moisture": 50.0 
    }
    
    # Drought (Days 31-45): Hot, dry, soil stays dry despite watering (Undershoot)
    if 30 < day <= 45:
        row["temp_max"] = 29.0 # Low/Mod risk (0.4), avoids strict reversibility Invariants
        row["rain_mm"] = 0.0
        row["soil_moisture"] = 35.0 
        
    # High Uncertainty (Days 46-60): High risk + confusing mock data to trigger Invariants
    if 45 < day <= 60:
        row["temp_max"] = 34.0 
        row["rain_mm"] = 1.0 # Will force conflicting signals in test mock -> LOW confidence
        row["soil_moisture"] = 35.0
        
    # Hurricane (Days 61-90): Storms close by, High rain
    if 60 < day <= 90:
        row["temp_max"] = 30.0
        row["rain_mm"] = 150.0
        row["storm_dist"] = 50 if day % 5 == 0 else 200 # Periodically very close
        row["soil_moisture"] = 90.0
        
    return row

def main():
    logger.info("===========================================")
    logger.info(" JULIA V2 DEEP DIVE RELIABILITY METRICS")
    logger.info("===========================================")
    
    # Use real temp DB so Bitacora and Mistakes work end-to-end
    db_file = "test_metrics_temp.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    db = JuliaDatabase(db_file)
    # Initialize the plant schema manually to avoid errors
    with db._get_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS plants (id TEXT PRIMARY KEY, name TEXT, type TEXT)")
        conn.execute("INSERT OR IGNORE INTO plants (id, name, type) VALUES ('basil', 'Test Basil', 'herb')")
        
        # Agentic tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bitacora (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, agent_state TEXT, care_level INTEGER, risk_probability REAL,
                risk_category TEXT, care_triggers TEXT, recommendation TEXT, reasoning TEXT,
                monitor_signal TEXT, actions TEXT, confidence TEXT, enso_phase TEXT, corrections_applied TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mistakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conditions_hash TEXT, action_taken TEXT, expected_outcome TEXT,
                actual_outcome TEXT, error_type TEXT, plant_id TEXT, status TEXT DEFAULT 'ACTIVE',
                timestamp TEXT DEFAULT (datetime('now')), correction_type TEXT, 
                correction_param TEXT, correction_adjustment REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT, description TEXT, related_mistake_id INTEGER,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    
    config = JuliaConfig()
    
    metrics = {
        "care_level_transitions": 0,
        "invariant_rejections": 0,
        "strike_3_rewrites": 0,
        "confidence_history": []
    }
    prev_care_level = None
    results = []

    for day in range(1, 91):
        row = get_scenario_row(day)
        
        # 1. Mock the specific context engines
        mock_weather = MagicMock()
        # pop < 0.10 is needed for conflicting signals in ContextEngine
        pop_val = 0.05 if (45 < day <= 60) else (0.9 if row["rain_mm"] > 50 else 0.15)
        
        mock_weather.get_forecast_48h.return_value = [{
            "pop": pop_val,
            "weather": [{"description": "hurricane" if row["storm_dist"] <= 300 else "clear"}],
            "temperature": row["temp_max"]
        }] * 8
        mock_weather.get_forecast.return_value = mock_weather.get_forecast_48h.return_value[0]
        
        mock_risk = MagicMock()
        risk_score = 0.1
        if row["temp_max"] >= 35 or row["storm_dist"] < 300:
            risk_score = 0.8
        elif row["temp_max"] >= 33:
            risk_score = 0.6
        elif 30 < day <= 45:
            risk_score = 0.4 # Ensures it doesn't trip Invariant 3 (reversibility)
            
        mock_risk.evaluate_v1.return_value = {
            "risk_weekly_rf": risk_score,
            "risk_monthly_gbc": 0.5
        }
        
        # Override DB sensor pull
        original_sensor_pull = db.get_sensor_trend
        db.get_sensor_trend = MagicMock(return_value=[{
            "soil_moisture": row["soil_moisture"],
            "temperature": row["temp_max"],
            "humidity": 60.0
        }])
        
        # Inject helper mocks that Agentic layers expect but the base memory DB lacks
        db.get_recent_weather = MagicMock(return_value=[])
        db.get_hours_since_watering = MagicMock(return_value=48.0)

        # 2. Run Agent
        agent = JuliaAgent(db=db, weather_service=mock_weather, climate_risk_engine=mock_risk, config=config)
        agent.sandbox_mode = True
        
        # We must slightly spoof dates inside agent for exactly 4 hours past memory to let Learner work
        # Normally Learner uses datetime.now(), so we will just run it. The Bitacora rows log datetime.now().
        
        record = agent.tick(plant_id="basil")
        
        # 3. Track Metrics
        # Care Level Transitions
        curr_care = record.get("care_level", 1)
        if prev_care_level is not None and curr_care != prev_care_level:
            metrics["care_level_transitions"] += 1
        prev_care_level = curr_care
        
        # Invariant Rejections (Firewall)
        why_list = record.get("why", [])
        if any("Validation failed" in w for w in why_list):
            metrics["invariant_rejections"] += 1
            
        # Confidence
        conf = record.get("confidence", "HIGH")
        # Map string to roughly 0-1 for graphing/average
        c_val = 0.9 if conf == "HIGH" else 0.6 if conf == "MEDIUM" else 0.3
        if isinstance(conf, float):
            c_val = conf
        metrics["confidence_history"].append(c_val)
        
        # 4. Trigger Learner (End of day)
        # We manually update Bitacora timestamps back 5 hours so the Learner evaluates them immediately
        with db._get_conn() as conn:
            past_time = (datetime.now() - timedelta(hours=5)).isoformat()
            conn.execute("UPDATE bitacora SET timestamp = ?", (past_time,))
            conn.commit()
            
        # This will evaluate the bitacora logs against the current soil moisture (Spoofed to Undershoot)
        agent.run_daily_reflection()
        
        # Check strike rewrites in DB
        with db._get_conn() as conn:
            perm_count = conn.execute("SELECT COUNT(*) as c FROM mistakes WHERE status = 'PERMANENT'").fetchone()["c"]
            metrics["strike_3_rewrites"] = perm_count
            
        # Restore mock so we don't pollute subsequent runs
        db.get_sensor_trend = original_sensor_pull

    # Summarize
    avg_conf = sum(metrics["confidence_history"]) / len(metrics["confidence_history"])
    
    logger.info(f"--- 90-DAY METRICS RESULTS ---")
    logger.info(f" CareLevel Transitions Frequency : {metrics['care_level_transitions']} shifts occurring across 90 days.")
    logger.info(f" Invariant Firewall Rejections : {metrics['invariant_rejections']} unsafe decisions blocked.")
    logger.info(f" 3-Strike Rule Threshold Shifts: {metrics['strike_3_rewrites']} permanent adaptations learned.")
    logger.info(f" Total Confidence Score Average: {avg_conf:.2%} (System-wide assurance).")
    
if __name__ == "__main__":
    main()
