import pytest
from datetime import datetime

from julia.agentic.state import WorldState, AgenticContext
from julia.agentic.caring import CaringLayer, AgentState

class MockMistakes:
    def __init__(self, failure=False):
        self.failure = failure
    def has_similar_failure(self, h):
        return self.failure

@pytest.fixture
def caring():
    return CaringLayer()

def test_determine_state_storm_prep(caring):
    # Risk > 0.6 triggers STORM_PREP
    ws = WorldState(risk_weekly=0.65)
    state = caring.determine_state(ws)
    assert state == AgentState.STORM_PREP

    # Proximity < 500 triggers STORM_PREP
    ws = WorldState(storm_proximity_km=400, risk_weekly=0.1)
    state = caring.determine_state(ws)
    assert state == AgentState.STORM_PREP

def test_determine_state_recovery(caring):
    # Hours since storm < 120 triggers RECOVERY
    ws = WorldState(hours_since_last_storm=24)
    state = caring.determine_state(ws)
    assert state == AgentState.RECOVERY

    # Hours since drought < 72 triggers RECOVERY
    ws = WorldState(hours_since_drought_end=48)
    state = caring.determine_state(ws)
    assert state == AgentState.RECOVERY

def test_determine_state_supportive(caring):
    # Validates SUPPORTIVE when past mistake exists
    ws = WorldState()
    state = caring.determine_state(ws, past_mistakes=MockMistakes(failure=True))
    assert state == AgentState.SUPPORTIVE
    
    # Risk > 0.4 triggers SUPPORTIVE
    ws = WorldState(risk_weekly=0.45)
    state = caring.determine_state(ws)
    assert state == AgentState.SUPPORTIVE

def test_determine_state_normal(caring):
    ws = WorldState()
    state = caring.determine_state(ws)
    assert state == AgentState.NORMAL

def test_calculate_care_level(caring):
    ws = WorldState(risk_weekly=0.5) # risk score = 2
    ctx = AgenticContext(world_state=ws, past_mistakes=MockMistakes(failure=False), user_uncertainty_detected=False)
    level = caring.calculate_care_level(ctx)
    assert level == 2
    
    # Maxes out at 3
    ws = WorldState(risk_weekly=0.8) # risk score = 3
    ctx = AgenticContext(world_state=ws, past_mistakes=MockMistakes(failure=True), user_uncertainty_detected=True)
    level = caring.calculate_care_level(ctx)
    assert level == 3
    
    # Minimum is 1 if risk > 0
    ws = WorldState(risk_weekly=0.1) # risk score = 0
    ctx = AgenticContext(world_state=ws, past_mistakes=MockMistakes(failure=False))
    level = caring.calculate_care_level(ctx)
    assert level == 1
