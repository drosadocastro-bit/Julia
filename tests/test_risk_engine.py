import pytest
from julia.core.risk_engine import ClimateRiskEngine

@pytest.fixture
def engine():
    return ClimateRiskEngine()

def test_clamp(engine):
    assert engine.clamp(-1.0) == 0.0
    assert engine.clamp(0.5) == 0.5
    assert engine.clamp(2.0) == 1.0

def test_no_risk_scenario(engine):
    # No storms, normal rain, neutral ENSO, normal daylight
    result = engine.evaluate(
        storm_dist_km=9999,
        rain_anomaly_pct=0,
        drought_idx=0,
        day_length_min=720,
        annual_mean_min=720,
        enso_phase=0
    )
    
    assert result["composite"]["final_risk"] == 0.0
    assert result["composite"]["category"] == "LOW"
    assert result["subscores"]["storm_risk"] == 0.0
    assert result["subscores"]["rainfall_risk"] == 0.0
    assert result["subscores"]["enso_modifier"] == 0.0

def test_active_storm_scenario(engine):
    # Active storm 100km away, cat 4 (130mph)
    result = engine.evaluate(
        storm_dist_km=100,
        storm_vmax=130,
        storm_count=2,
        storm_days_since=1,
        # Rest normal
        rain_anomaly_pct=0,
        drought_idx=0,
        day_length_min=720,
        annual_mean_min=720,
        enso_phase=0
    )
    
    # Base expected:
    # dist factor ~ e^(-100/150) = 0.513
    # v max = 130/150 = 0.86
    # count = 2/3 = 0.66
    # base = (0.5 * 0.513) + (0.3 * 0.86) + (0.2 * 0.66) = 0.256 + 0.258 + 0.132 = 0.646
    # recency = 1 - 1/7 = 0.857
    # storm_risk ~ 0.55
    # final ~ 0.40 * 0.55 = 0.22 (LOW/MODERATE boundary)
    
    assert result["subscores"]["storm_risk"] > 0.4
    assert result["subscores"]["storm_risk"] < 0.7
    assert result["composite"]["category"] in ["LOW", "MODERATE"] 

def test_severe_drought_scenario(engine):
    # Severe drought, El Niño, long daylight
    result = engine.evaluate(
        storm_dist_km=9999,
        rain_anomaly_pct=-60, # 60% deficit
        drought_idx=-2.5,
        day_length_min=800,
        annual_mean_min=720,
        enso_phase=1 # El Nino
    )
    
    assert result["subscores"]["drought_risk"] > 0.8  # (2.5/3.0) * 1.15 multiplier
    assert result["subscores"]["evap_risk"] > 0.5     # Amplified by El nino and deficit
    assert result["composite"]["final_risk"] > 0.4
    assert result["composite"]["category"] in ["MODERATE", "HIGH"]

def test_la_nina_flood_scenario(engine):
    # La Nina, huge rain excess
    result = engine.evaluate(
        storm_dist_km=9999,
        rain_anomaly_pct=80, # 80% excess
        drought_idx=0,
        day_length_min=720,
        annual_mean_min=720,
        enso_phase=-1 # La Nina
    )
    
    # rain risk should be 1.0 clamped
    assert result["subscores"]["rainfall_risk"] == 1.0
    assert result["composite"]["final_risk"] > 0.25
    assert result["composite"]["category"] in ["MODERATE", "HIGH"]

def test_max_catastrophe_scenario(engine):
    # Everything is terrible
    result = engine.evaluate(
        storm_dist_km=0,
        storm_vmax=200,
        storm_count=5,
        storm_days_since=0,
        rain_anomaly_pct=-100,
        drought_idx=-5,
        day_length_min=900,
        annual_mean_min=720,
        enso_phase=1
    )
    
    assert result["subscores"]["storm_risk"] == 1.0
    assert result["subscores"]["drought_risk"] == 1.0
    # Final risk maxes out the scale
    assert result["composite"]["final_risk"] == 1.0
    assert result["composite"]["category"] == "CRITICAL"
