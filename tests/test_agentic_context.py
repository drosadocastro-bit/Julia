import pytest
from datetime import datetime

from julia.agentic.state import WorldState, AgenticContext
from julia.agentic.perception import PerceptionLayer
from julia.agentic.context_engine import ContextEngine
from julia.agentic.caring import AgentState

# Mocks
class MockWeather:
    def get_forecast_48h(self):
        return [
            {"pop": 0.9, "weather": [{"description": "heavy tropical storm"}]},
            {"pop": 0.8, "weather": [{"description": "rain"}]}
        ]

class MockRisk:
    def evaluate_v1(self, features):
        return {"risk_weekly_rf": 0.75, "risk_monthly_gbc": 0.4}

class MockDB:
    def get_decision_history(self, days):
        return [{"action": "watered"}]
    def get_recent_weather(self, hours):
        return []

class MockMemory:
    def get_mistakes(self, conditions_hash):
        return [{"id": 1}, {"id": 2}, {"id": 3}] # 3 past mistakes

@pytest.fixture
def perception():
    return PerceptionLayer(db=None, weather_service=MockWeather(), risk_engine=MockRisk())

@pytest.fixture
def context_engine():
    return ContextEngine(db=MockDB(), memory=MockMemory())

def test_perception_layer_fuses_data(perception):
    state = perception.get_world_state()
    
    # Weather mock sets 90% rain and active tropical storm
    assert state.rain_probability_24h == 90.0
    assert state.disturbance_active is True
    
    # Risk mock sets 0.75
    assert state.risk_weekly == 0.75
    
    # Time context
    assert isinstance(state.timestamp, datetime)
    assert state.hurricane_season in [True, False] # depends on current month

def test_context_engine_builds_context(context_engine):
    state = WorldState(
        risk_weekly=0.8, 
        rain_probability_24h=90, 
        drought_active=False
    )
    
    ctx = context_engine.build_context(state)
    
    # Since risk is 0.8, CaringLayer should set state to STORM_PREP
    assert ctx.agent_state == AgentState.STORM_PREP
    
    # With risk 0.8 + 3 mistakes from mock, care level should hit 3
    assert ctx.care_level == 3
    
    # High risk + high rain = aligned signals
    # BUT 3 past mistakes = minus 1 aligned signal
    # Net: 0 aligned signals -> MODERATE
    assert ctx.confidence == "MODERATE (0.7)"
    
    # 3 mistakes flag error-prone pattern
    assert "Historically Error-Prone Condition" in ctx.detected_patterns

def test_context_confidence_low(context_engine):
    # Conflicting signals: extreme risk but no rain forecast, plus wet soil in drought
    state = WorldState(
        risk_weekly=0.9, 
        rain_probability_24h=5.0,
        drought_active=True,
        soil_moisture={"basil": 85.0} # extremely wet during drought
    )
    
    # Memory returns 0 mistakes by default if we don't mock it heavily, 
    # but the conflicting signals flag directly returns "LOW"
    # Wait, our MockMemory always returns 3 mistakes, which lowers score further.
    ctx = context_engine.build_context(state)
    assert ctx.confidence == "LOW (0.4)"
