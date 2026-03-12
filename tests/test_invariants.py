import pytest
from julia.agentic.invariants import Action, Recommendation, CaringInvariants

@pytest.fixture
def invariants():
    return CaringInvariants()

def test_has_why(invariants):
    act = Action(type="WATER", reversible=False)
    # Fails - no reasoning
    rec1 = Recommendation(action=act, monitor_signal="Check leaves")
    assert not invariants.has_why(rec1)
    
    # Passes - has reasoning
    rec2 = Recommendation(action=act, reasoning="Soil is very dry", monitor_signal="Check leaves")
    assert invariants.has_why(rec2)

def test_has_monitor_signal(invariants):
    act = Action(type="WATER", reversible=False)
    # Fails - no monitor signal
    rec1 = Recommendation(action=act, reasoning="Soil is very dry")
    assert not invariants.has_monitor_signal(rec1)
    
    # Passes
    rec2 = Recommendation(action=act, reasoning="Soil is dry", monitor_signal="Watch for leaf curl")
    assert invariants.has_monitor_signal(rec2)

def test_respects_reversibility(invariants):
    act_irreversible = Action(type="WATER", reversible=False)
    act_reversible = Action(type="ALERT_USER", reversible=True)
    
    # High risk (0.6) + Irreversible action = FAIL
    rec1 = Recommendation(
        action=act_irreversible, 
        reasoning="Dry", 
        monitor_signal="Watch",
        context_risk=0.6
    )
    assert not invariants.respects_reversibility(rec1)
    
    # High risk + Reversible = PASS
    rec2 = Recommendation(
        action=act_reversible, 
        reasoning="Dry", 
        monitor_signal="Watch",
        context_risk=0.6
    )
    assert invariants.respects_reversibility(rec2)
    
    # Low risk + Irreversible = PASS
    rec3 = Recommendation(
        action=act_irreversible, 
        reasoning="Dry", 
        monitor_signal="Watch",
        context_risk=0.2
    )
    assert invariants.respects_reversibility(rec3)

def test_handles_uncertainty(invariants):
    act = Action(type="WATER", reversible=False)
    
    # High uncertainty + No clarification = FAIL
    rec1 = Recommendation(
        action=act,
        reasoning="Might be dry?",
        monitor_signal="Watch",
        uncertainty=0.7,
        asks_clarification=False
    )
    assert not invariants.handles_uncertainty(rec1)
    
    # High uncertainty + Clarification asked = PASS
    rec2 = Recommendation(
        action=act,
        reasoning="Might be dry?",
        monitor_signal="Watch",
        uncertainty=0.7,
        asks_clarification=True
    )
    assert invariants.handles_uncertainty(rec2)

def test_overall_validation(invariants):
    # This should pass all invariants
    rec = Recommendation(
        action=Action(type="WATER", reversible=False),
        reasoning="Moisture at 40%, need to water.",
        monitor_signal="Check moisture in 4 hours.",
        context_risk=0.2, # Low risk means irreversible is ok
        uncertainty=0.1,  # Low uncertainty
        asks_clarification=False
    )
    assert invariants.validate(rec)
