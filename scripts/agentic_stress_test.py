import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from stress_test_engine import (
    generate_scenario_1_extreme_storms,
    generate_scenario_2_flash_drought,
    generate_scenario_3_la_nina_wet,
    generate_scenario_4_false_positives,
    EngineState
)

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from julia.agent import JuliaAgent
from julia.core.config import JuliaConfig

logger = logging.getLogger("agentic_stress_test")

def create_mock_environment(scenario_row, engine_state):
    """Mocks the 3 core integrations based on the current step in the scenario."""
    feats = engine_state.get_features()
    
    # 1. DB Mock (Sensors & History)
    mock_db = MagicMock()
    # Simple sensor physics based on rain
    moisture = 50.0 + (scenario_row["rain_mm"] * 5) - (scenario_row["temp_max"] - 30) * 2
    moisture = max(20.0, min(100.0, moisture))
    
    mock_db.get_sensor_trend.return_value = [{
        "soil_moisture": moisture,
        "temperature": scenario_row["temp_max"],
        "humidity": 60.0
    }]
    mock_db.get_decision_history.return_value = []
    mock_db.get_recent_weather.return_value = []
    mock_db.get_hours_since_watering.return_value = 48.0
    mock_db._get_conn.return_value.__enter__.return_value = MagicMock()

    # 2. WeatherService Mock
    mock_weather = MagicMock()
    mock_weather.get_forecast_48h.return_value = [{
        "pop": 0.9 if scenario_row["rain_mm"] > 20 else 0.1,
        "weather": [{"description": "hurricane" if scenario_row["storm_dist"] < 300 else "clear"}],
        "temperature": scenario_row["temp_max"]
    }] * 8

    # 3. ClimateRiskEngine Mock (We can use the real one, but mocking is faster for testing the pipeline)
    mock_risk = MagicMock()
    
    # Heuristic scoring to trigger the correct pipeline paths
    weekly_risk = 0.1
    if scenario_row["storm_dist"] < 300:
        weekly_risk = 0.9 # CRITICAL
    elif feats["rainfall_anomaly_percent"] < -50:
        weekly_risk = 0.8 # CRITICAL
        
    mock_risk.evaluate_v1.return_value = {
        "risk_weekly_rf": weekly_risk,
        "risk_monthly_gbc": 0.5
    }

    return mock_db, mock_weather, mock_risk

def run_agentic_simulation(df: pd.DataFrame, scenario_name: str):
    logger.info(f"--- Running Agentic Loop on {scenario_name} ---")
    
    state = EngineState()
    results = []
    
    # We create a new agent per scenario run to reset state
    config = JuliaConfig()
    
    for _, row in df.iterrows():
        state.add_day(row["rain_mm"], row["storm_dist"], row["storm_vmax"])
        
        # Inject the scenario into the mock wrappers
        db, weather, risk = create_mock_environment(row, state)
        
        # Instantiate Julia
        agent = JuliaAgent(db=db, weather_service=weather, climate_risk_engine=risk, config=config)
        agent.sandbox_mode = True 
        
        # TICK!
        record = agent.tick(plant_id="basil")
        
        results.append({
            "day": len(results) + 1,
            "date": row["date"].strftime("%Y-%m-%d"),
            "rain_mm": row["rain_mm"],
            "care_level": record.get("care_level", 1),
            "state": record.get("state", "UNKNOWN"),
            "primary_action": record["actions"][0]["type"] if "actions" in record and record["actions"] else "NONE",
            "risk_category": record.get("risk_category", "LOW")
        })
        
    res_df = pd.DataFrame(results)
    
    # diagnostics
    panics = (res_df["state"] == "RECOVERY").sum()
    lockouts = (res_df["primary_action"] == "EMERGENCY_LOCKOUT").sum()
    
    logger.info(f" > System Panics (Crash Fallbacks): {panics}")
    logger.info(f" > Emergency Lockouts Triggered: {lockouts}")
    
    return res_df

def run():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("====================================")
    logger.info(" JULIA V2 AGENTIC PIPELINE HARNESS")
    logger.info("====================================")
    
    scenarios = [
        generate_scenario_1_extreme_storms(),
        generate_scenario_2_flash_drought()
    ]
    
    all_results = []
    for payload, name in scenarios:
        res = run_agentic_simulation(payload, name)
        res["scenario"] = name
        all_results.append(res)
        
    final_df = pd.concat(all_results, ignore_index=True)
    out_path = Path(__file__).parent.parent / "julia" / "logs" / "agentic_stress_test_results.csv"
    final_df.to_csv(out_path, index=False)
    logger.info(f"Saved detailed diagnostics to {out_path}")

if __name__ == "__main__":
    run()
