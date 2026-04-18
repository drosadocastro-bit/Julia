import os
import sys
import json
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from julia.agent import JuliaAgent
from julia.agentic.perception import WorldState
from julia.agentic.bitacora import Bitacora
from julia.core.database import JuliaDatabase
from julia.core.config import JuliaConfig
from julia.core.decision_engine import SensorData, WeatherForecast

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("HarvestOracle")

class BiologicalModel:
    """
    Simulates the biological health of a plant over time.
    """
    def __init__(self, name="Basil"):
        self.name = name
        self.health = 50.0  # 0 to 100
        self.history = []
        
    def step(self, temp, water_ml, storm_dist, is_drought, has_storm_prep):
        # 1. Temperature Stress
        if temp > 32.0:
            if water_ml < 100:
                self.health -= 5.0 # Heat stress without enough water
            else:
                self.health -= 1.0 # Managed heat
        elif 22.0 <= temp <= 28.0:
            self.health += 2.0 # Optimal growth
            
        # 2. Storm Impact
        if storm_dist < 100:
            if not has_storm_prep:
                self.health -= 40.0 # Catastrophic damage
            else:
                self.health -= 10.0 # Mitigated damage
                
        # 3. Drought Impact
        if is_drought and water_ml == 0:
            self.health -= 8.0
            
        # 4. Oversaturation
        if water_ml > 500:
            self.health -= 2.0 # Root rot risk
            
        # Bounds
        self.health = max(0.0, min(100.0, self.health))
        self.history.append(self.health)
        
    def get_status(self):
        if self.health > 90: return "🌿 GREAT HARVEST"
        if self.health > 60: return "🥗 MEDIUM YIELD"
        if self.health > 30: return "🥀 LOW YIELD"
        return "💀 CROP FAILURE"

def run_harvest_sim(scenario_name="CATASTROPHE", days=180):
    logger.info(f"--- STARTING HARVEST ORACLE: {scenario_name} ---")
    
    # 1. Setup Mock DB
    db_path = f"harvest_{scenario_name.lower()}.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    db = JuliaDatabase(db_path=db_path)
    # Inject Agentic Tables
    with db._get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bitacora (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, agent_state TEXT, care_level INTEGER, 
                risk_probability REAL, risk_category TEXT, 
                care_triggers TEXT, recommendation TEXT, reasoning TEXT, 
                monitor_signal TEXT, actions TEXT, 
                confidence TEXT, enso_phase TEXT, corrections_applied TEXT
            );
            CREATE TABLE IF NOT EXISTS mistakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conditions_hash TEXT, plant_id TEXT, action_taken TEXT, expected_outcome TEXT,
                actual_outcome TEXT, error_type TEXT, status TEXT DEFAULT 'ACTIVE',
                timestamp TEXT DEFAULT (datetime('now')), correction_type TEXT, 
                correction_param TEXT, correction_adjustment REAL
            );
            CREATE TABLE IF NOT EXISTS plants (
                id TEXT PRIMARY KEY, name TEXT, emoji TEXT, min_moisture REAL, max_moisture REAL
            );
            INSERT INTO plants (id, name, emoji, min_moisture, max_moisture)
            VALUES ('basil', 'Basil', '🌿', 40, 70);
        """)

    # 2. Setup Agent
    mock_weather = MagicMock()
    mock_risk = MagicMock()
    config = JuliaConfig()
    agent = JuliaAgent(db=db, weather_service=mock_weather, climate_risk_engine=mock_risk, config=config)
    agent.executor.autonomous_mode = True # We want to see what she WOULD do
    
    plant = BiologicalModel("Basil")
    
    results = []
    
    for day in range(days):
        # Define Scenario Conditions
        temp = 28.0
        storm_dist = 9999
        risk = 0.1
        is_drought = False
        moisture = 55.0
        
        if scenario_name == "CATASTROPHE":
            if 30 <= day <= 60: # Extended Drought in seedling phase
                temp = 36.0
                is_drought = True
                moisture = 32.0
            if 100 <= day <= 110: # Late Hurricane
                storm_dist = 40
                risk = 0.98
        elif scenario_name == "GOLDEN":
            temp = 25.0
            moisture = 55.0
        else: # REALISTIC
            if day % 15 == 0: temp = 33.0
            if day % 25 == 0: storm_dist = 150
            moisture = 40.0 + (day % 15)
            risk = 0.3 if storm_dist < 500 else 0.1
            
        # Mock Perceptions
        with patch.object(agent.perception, 'get_world_state') as mock_gather:
            mock_ws = WorldState(
                timestamp=datetime(2026, 1, 1) + timedelta(days=day),
                temperature=temp,
                humidity=70.0,
                soil_moisture={"basil": moisture},
                rain_probability_24h=risk * 100,
                risk_weekly=risk,
                drought_active=is_drought,
                storm_proximity_km=storm_dist
            )
            mock_gather.return_value = mock_ws
            
            # Execute agents loop
            record = agent.tick(plant_id="basil")
            
            # Extract action
            actions = record.get("actions", [])
            water_ml = sum(a.get("amount_ml", 0) for a in actions if a.get("type") == "WATER")
            has_prep = any(a.get("type") == "STORM_PREP" or record.get("agent_state") == "STORM_PREP" for a in actions)
            if record.get("agent_state") == "STORM_PREP": has_prep = True
            
            # Biological Step
            plant.step(temp, water_ml, storm_dist, is_drought, has_prep)
            
            results.append({
                "day": day,
                "health": plant.health,
                "water_ml": water_ml,
                "state": record.get("agent_state")
            })
            
        if day % 7 == 0:
            agent.run_daily_reflection()

    # Final Report
    logger.info(f"FINAL HEALTH: {plant.health:.1f} ({plant.get_status()})")
    
    df = pd.DataFrame(results)
    df.to_csv(f"harvest_results_{scenario_name.lower()}.csv", index=False)
    return plant.health, plant.get_status()

if __name__ == "__main__":
    run_harvest_sim("CATASTROPHE")
    run_harvest_sim("GOLDEN")
    run_harvest_sim("REALISTIC")
