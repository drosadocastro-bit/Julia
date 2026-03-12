import pytest
from julia.agentic.state import WorldState, AgenticContext
from julia.agentic.caring import AgentState
from julia.agentic.invariants import ActionPlan, Action
from julia.agentic.planner import AgenticPlanner

class MockDecisionEngine:
    def decide(self, context):
        return {
            "should_water": True,
            "water_multiplier": 1.2,
            "reason": "Soil is dry.",
            "monitor_signal": "Watch leaf wilt."
        }

@pytest.fixture
def planner():
    return AgenticPlanner(decision_engine=MockDecisionEngine())

def test_planner_applies_care_level(planner):
    # Setup context with high care level
    state = WorldState(risk_weekly=0.5)
    ctx = AgenticContext(world_state=state, care_level=2, agent_state=AgentState.SUPPORTIVE)
    
    plan = planner.plan(ctx)
    
    # Care level 2 should enable these flags
    assert plan.break_into_steps is True
    assert plan.ask_confirmation is True
    
    # AgentState SUPPORTIVE should enable these
    assert plan.extra_explanation is True
    assert plan.gentler_tone is True

def test_planner_applies_instincts(planner):
    # Setup context for a storm (risk < 0.5 so we don't trigger the reversibility firewall)
    state = WorldState(risk_weekly=0.4, storm_proximity_km=300)
    ctx = AgenticContext(world_state=state, care_level=3, agent_state=AgentState.STORM_PREP)
    
    plan = planner.plan(ctx)
    
    # Verify instinct reasoning was added
    assert any("Imminent storm" in r for r in plan.reasoning)
    assert any("Reducing water" in r for r in plan.reasoning)
    
    # Verify water multiplier was halved from 1.2 to 0.6
    action = [a for a in plan.actions if a.type == "WATER"][0]
    assert action.multiplier == 0.6

def test_planner_firewall_fallback(planner):
    # Invariants firewall should catch an action that is irreversible but has high uncertainty
    
    class UnsafeDecisionEngine:
        def decide(self, context):
            return {
                "should_water": True,
                "water_multiplier": 1.0,
                "reason": "Guessing.",
            }
            
    unsafe_planner = AgenticPlanner(decision_engine=UnsafeDecisionEngine())
    state = WorldState()
    # High uncertainty but low care (no ask_confirmation flag) fails Invariant 4
    ctx = AgenticContext(world_state=state, care_level=1, agent_state=AgentState.NORMAL, confidence="LOW (0.4)")
    
    # Because there is no monitor_signal, the Invariants firewall will reject it.
    plan = unsafe_planner.plan(ctx)
    
    # Verify the fallback mechanism replaced the unsafe plan
    assert plan.actions[0].type == "REQUEST_GUIDANCE"
    assert "Validation failed" in plan.reasoning[0]

def test_planner_applies_autocorrect(planner):
    # Setup context with an active multiplier autocorrect from a past mistake
    past_mistakes = [{
        "status": "ACTIVE",
        "correction_type": "MULTIPLIER",
        "correction_adjustment": 0.8  # Reduce water by 20%
    }]
    
    state = WorldState()
    ctx = AgenticContext(
        world_state=state, 
        care_level=1, 
        agent_state=AgentState.NORMAL, 
        past_mistakes=past_mistakes
    )
    
    plan = planner.plan(ctx)
    
    # Base decision engine multiplier is 1.2
    # Autocorrect is 0.8
    # Final multiplier should be 1.2 * 0.8 = 0.96
    action = [a for a in plan.actions if a.type == "WATER"][0]
    assert action.multiplier == 0.96
    
    # Verify the reasoning traces the autocorrect
    assert any("Autocorrect: Applying 0.8x" in r for r in plan.reasoning)
