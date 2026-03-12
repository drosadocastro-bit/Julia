
import pytest
from julia.core.brain import Brain

@pytest.fixture
def brain():
    return Brain()

def test_init_brain(brain):
    assert brain.version == "v1-rules"

def test_hydrated_plant(brain):
    """Rule 1: Plant has enough water -> WAIT."""
    # Current 60, Min 50 -> OK
    decision = brain.decide("basil", 60.0, 50.0, 0.0)
    assert decision.action == "WAIT"
    assert "sufficient" in decision.reason

def test_emergency_water(brain):
    """Rule 2: Plant is critically dry -> WATER regardless of rain."""
    # Current 10, Min 50 (Critical = 40) -> WATER
    # Rain is 90% (but we can't wait!)
    decision = brain.decide("basil", 10.0, 50.0, 90.0)
    assert decision.action == "WATER"
    assert "CRITICAL" in decision.reason

def test_rain_delay(brain):
    """Rule 3: Plant is slightly dry, but rain is coming -> WAIT."""
    # Current 45, Min 50 (Not critical)
    # Rain 80% -> WAIT
    decision = brain.decide("basil", 45.0, 50.0, 80.0)
    assert decision.action == "WAIT"
    assert "Rain forecast" in decision.reason

def test_normal_water(brain):
    """Rule 4: Plant is dry, no rain -> WATER."""
    # Current 45, Min 50
    # Rain 10% -> WATER
    decision = brain.decide("basil", 45.0, 50.0, 10.0)
    assert decision.action == "WATER"
    assert "below target" in decision.reason
