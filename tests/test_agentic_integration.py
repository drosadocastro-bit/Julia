import pytest
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import MagicMock

from julia.core.config import JuliaConfig
from julia.agent import JuliaAgent

@pytest.fixture
def mock_db():
    db = MagicMock()
    # Mock for PerceptionLayer
    db.get_recent_weather.return_value = {
        "temperature": 34.0,
        "humidity": 45.0,
        "timestamp": datetime.now().isoformat()
    }
    db.get_sensor_trend.return_value = [{"soil_moisture": 30.0}] # Dry soil
    
    # Mock for ContextEngine
    db.get_decision_history.return_value = []
    
    # Mock for DecisionEngine
    db.get_hours_since_watering.return_value = 48.0
    
    # Mock for Learner 
    db._get_conn.return_value.__enter__.return_value = MagicMock()
    
    return db

@pytest.fixture
def mock_weather():
    wm = MagicMock()
    wm.get_forecast.return_value = {
        "rain_probability_24h": 0.0,
        "rain_probability_48h": 0.0,
        "temp_high": 35.0,
        "temp_low": 28.0,
        "humidity": 45.0,
        "description": "Hot and dry",
        "is_available": True
    }
    return wm
    
@pytest.fixture
def mock_risk():
    rm = MagicMock()
    # Correcting method call used by PerceptionLayer
    rm.evaluate_v1.return_value = {
        "risk_weekly_rf": 0.85, # CRITICAL risk
        "risk_monthly_gbc": 0.6
    }
    return rm

def test_full_ooda_loop_integration(mock_db, mock_weather, mock_risk):
    config = JuliaConfig()
    
    # Needs a mock DecisionEngine for the planner
    agent = JuliaAgent(db=mock_db, weather_service=mock_weather, climate_risk_engine=mock_risk, config=config)
    agent.sandbox_mode = True # No actual hardware calls
    
    # Execute the tick
    record = agent.tick(plant_id="basil")
    
    # The record should show that the extreme heat triggering CRITICAL risk
    # triggered the Phase 12 EMERGENCY_LOCKOUT rule in the AgenticPlanner 
    # (because risk >= 0.75).
    
    assert record["care_level"] >= 1
    assert record["risk_category"] == "CRITICAL"
    
    # The Invariant firewall and Planner overrides should force this to EMERGENCY_LOCKOUT
    assert any(a["type"] == "EMERGENCY_LOCKOUT" for a in record["actions"])
    assert any("CRITICAL weather risk" in r for r in record["why"])

def test_daily_reflection(mock_db, mock_weather, mock_risk):
    agent = JuliaAgent(db=mock_db, weather_service=mock_weather, climate_risk_engine=mock_risk, config=JuliaConfig())
    # Should run without throwing an exception
    agent.run_daily_reflection()
