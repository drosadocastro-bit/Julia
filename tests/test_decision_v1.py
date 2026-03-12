import pytest
from julia.core.config import JuliaConfig, PlantProfile, WeatherConfig
from julia.core.decision_engine import JuliaDecisionEngine, WaterDecision, WeatherForecast
from julia.sensors.sensor_reader import SensorData

@pytest.fixture
def config():
    c = JuliaConfig()
    c.weather = WeatherConfig(rain_skip_threshold=80)
    # Mock a plant profile
    c._profiles = {"test_plant": PlantProfile(
        name="Test Plant",
        emoji="🌱",
        min_moisture=30,
        max_moisture=70,
        water_amount_ml=200,
        min_hours_between_watering=12
    )}
    return c

@pytest.fixture
def engine(config):
    return JuliaDecisionEngine(config)

def test_ml_critical_override_forces_emergency_watering(engine):
    # Soil is dry (20 < 30) - normal watering would happen
    # But with WEEKLY_CRITICAL, amount should be 1.3x and bypass cooldown
    sensor = SensorData(soil_moisture=20, temperature=25, humidity=50, sensor_id="test_plant", timestamp="2026-01-01T12:00:00Z")
    weather = WeatherForecast(rain_probability_24h=10)
    
    # Mock ML Risk output
    climate_risk = {
        "composite": {
            "category": "CRITICAL",
            "final_advisory": "WEEKLY_CRITICAL_OVERRIDE"
        }
    }
    
    result = engine.decide("test_plant", sensor, weather, plant_health=None, climate_risk=climate_risk)
    
    # Normally, WaterDecision.WATER_NOW. Under Critical ML override -> EMERGENCY (since it bypasses cooldown)
    assert result.decision == WaterDecision.EMERGENCY
    assert result.water_amount_ml == int(200 * 1.3) # 260ml
    assert "ML Advisory: WEEKLY_CRITICAL_OVERRIDE" in result.reason
    assert "ML Risk CRITICAL: +30% water" in result.reason

def test_ml_advisory_no_action(engine):
    # Soil is perfectly fine (50) - normal NO_ACTION
    # Monthly advisory should append warning but not force watering
    sensor = SensorData(soil_moisture=50, temperature=25, humidity=50, sensor_id="test_plant", timestamp="2026-01-01T12:00:00Z")
    weather = WeatherForecast(rain_probability_24h=10)
    
    climate_risk = {
        "composite": {
            "category": "HIGH",
            "final_advisory": "MONTHLY_ADVISORY"
        }
    }
    
    result = engine.decide("test_plant", sensor, weather, plant_health=None, climate_risk=climate_risk)
    
    assert result.decision == WaterDecision.NO_ACTION
    assert result.water_amount_ml == 0
    assert "ML Advisory: MONTHLY_ADVISORY" in result.reason
