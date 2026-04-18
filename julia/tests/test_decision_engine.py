"""
Tests for julia.core.decision_engine — The brain of Julia.

Covers all 7 decision paths:
1. EMERGENCY — wilting + dry soil
2. SKIP_INVALID — bad sensor data
3. SKIP_RAIN — rain forecast
4. SKIP_WET — soil too wet
5. SKIP_RECENT — watered too recently
6. WATER_NOW — soil is dry
7. NO_ACTION — everything is fine

Also tests:
- Temperature adjustment for hot days
- Cooldown tracking
- Edge cases (values at exact thresholds)
"""

import unittest
from datetime import datetime, timezone, timedelta

from julia.core.config import JuliaConfig, PlantProfile
from julia.core.decision_engine import (
    JuliaDecisionEngine,
    WaterDecision,
    WeatherForecast,
    PlantHealth,
    WateringResult,
)
from julia.sensors.sensor_reader import SensorData


def _make_sensor(moisture=50.0, temp=28.0, humidity=70.0, valid=True):
    """Helper to create test SensorData."""
    return SensorData(
        soil_moisture=moisture,
        temperature=temp,
        humidity=humidity,
        sensor_id="test",
        timestamp=datetime.now(timezone.utc).isoformat(),
        is_valid=valid,
    )


def _make_weather(rain_24h=0.0, temp_high=30.0, available=True):
    """Helper to create test WeatherForecast."""
    return WeatherForecast(
        rain_probability_24h=rain_24h,
        rain_probability_48h=rain_24h * 0.8,
        temp_high=temp_high,
        temp_low=22.0,
        humidity=70.0,
        description="Test weather",
        is_available=available,
    )


class TestDecisionEngine(unittest.TestCase):
    """Test all decision paths in the Julia Decision Engine."""

    def setUp(self):
        self.config = JuliaConfig()
        self.engine = JuliaDecisionEngine(self.config)

    # ==========================================================
    # Priority 1: EMERGENCY
    # ==========================================================

    def test_emergency_wilting_dry_soil(self):
        """Wilting plant + dry soil = emergency watering at 1.5x."""
        sensor = _make_sensor(moisture=30.0)  # Below basil min (40)
        weather = _make_weather()
        health = PlantHealth(status="wilting", confidence=0.85)

        result = self.engine.decide("basil", sensor, weather, health)

        self.assertEqual(result.decision, WaterDecision.EMERGENCY)
        self.assertEqual(result.water_amount_ml, 300)  # 200 * 1.5
        self.assertIn("Emergency", result.reason)
        self.assertTrue(result.should_water())

    def test_no_emergency_wilting_wet_soil(self):
        """Wilting but soil is still wet — may be overwatering issue, don't emergency water."""
        sensor = _make_sensor(moisture=60.0)  # Above basil min
        weather = _make_weather()
        health = PlantHealth(status="wilting", confidence=0.85)

        result = self.engine.decide("basil", sensor, weather, health)

        # Should NOT be emergency — soil is fine, wilting from other cause
        self.assertNotEqual(result.decision, WaterDecision.EMERGENCY)

    def test_no_emergency_low_confidence(self):
        """Wilting detection below confidence threshold — don't trust it."""
        sensor = _make_sensor(moisture=30.0)
        weather = _make_weather()
        health = PlantHealth(status="wilting", confidence=0.4)  # Below 0.6

        result = self.engine.decide("basil", sensor, weather, health)

        self.assertNotEqual(result.decision, WaterDecision.EMERGENCY)

    # ==========================================================
    # Priority 2: INVALID SENSOR
    # ==========================================================

    def test_skip_invalid_sensor(self):
        """Invalid sensor data = skip watering (don't water blind)."""
        sensor = _make_sensor(moisture=-1.0, valid=False)
        weather = _make_weather()

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.SKIP_INVALID)
        self.assertEqual(result.water_amount_ml, 0)
        self.assertFalse(result.should_water())

    # ==========================================================
    # Priority 3: RAIN FORECAST
    # ==========================================================

    def test_skip_rain_high_probability(self):
        """Rain > 60% = skip watering."""
        sensor = _make_sensor(moisture=35.0)  # Dry, would normally water
        weather = _make_weather(rain_24h=75.0)

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.SKIP_RAIN)
        self.assertEqual(result.water_amount_ml, 0)
        self.assertIn("Rain", result.reason)

    def test_no_skip_rain_low_probability(self):
        """Rain < 60% = don't skip for rain."""
        sensor = _make_sensor(moisture=35.0)
        weather = _make_weather(rain_24h=30.0)

        result = self.engine.decide("basil", sensor, weather)

        self.assertNotEqual(result.decision, WaterDecision.SKIP_RAIN)

    def test_no_skip_rain_unavailable(self):
        """Weather unavailable = don't skip for rain (can't trust missing data)."""
        sensor = _make_sensor(moisture=35.0)
        weather = _make_weather(rain_24h=80.0, available=False)

        result = self.engine.decide("basil", sensor, weather)

        self.assertNotEqual(result.decision, WaterDecision.SKIP_RAIN)

    # ==========================================================
    # Priority 4: SOIL TOO WET
    # ==========================================================

    def test_skip_wet_soil(self):
        """Soil above max moisture = skip."""
        sensor = _make_sensor(moisture=80.0)  # Above basil max (70)
        weather = _make_weather()

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.SKIP_WET)
        self.assertEqual(result.water_amount_ml, 0)

    # ==========================================================
    # Priority 5: RECENT WATERING
    # ==========================================================

    def test_skip_recent_watering(self):
        """Watered recently within cooldown = skip."""
        sensor = _make_sensor(moisture=38.0)  # Dry
        weather = _make_weather()

        # Record watering 2 hours ago (basil min is 12h)
        self.engine.record_watering(
            "basil",
            datetime.now(timezone.utc) - timedelta(hours=2)
        )

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.SKIP_RECENT)

    def test_no_skip_after_cooldown(self):
        """Watered long ago — cooldown expired, should water if dry."""
        sensor = _make_sensor(moisture=35.0)
        weather = _make_weather()

        # Record watering 15 hours ago (basil min is 12h)
        self.engine.record_watering(
            "basil",
            datetime.now(timezone.utc) - timedelta(hours=15)
        )

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.WATER_NOW)

    # ==========================================================
    # Priority 6: WATER NOW
    # ==========================================================

    def test_water_dry_soil(self):
        """Soil below min = water."""
        sensor = _make_sensor(moisture=30.0)  # Below basil min (40)
        weather = _make_weather()

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.WATER_NOW)
        self.assertEqual(result.water_amount_ml, 200)  # Basil default
        self.assertTrue(result.should_water())

    def test_water_hot_day_adjustment(self):
        """Hot day (>30°C) = increase water amount."""
        sensor = _make_sensor(moisture=30.0, temp=35.0)  # 5 degrees above 30
        weather = _make_weather()

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.WATER_NOW)
        # 200 * (1 + 5 * 0.04) = 200 * 1.20 = 240
        self.assertEqual(result.water_amount_ml, 240)

    def test_water_very_hot_day_capped(self):
        """Extremely hot day — adjustment capped at 30%."""
        sensor = _make_sensor(moisture=30.0, temp=45.0)  # 15 degrees above 30
        weather = _make_weather()

        result = self.engine.decide("basil", sensor, weather)

        # 15 * 0.04 = 0.60, but capped at 0.30. So 200 * 1.30 = 260
        self.assertEqual(result.water_amount_ml, 260)

    # ==========================================================
    # Priority 7: NO ACTION
    # ==========================================================

    def test_no_action_happy_plant(self):
        """Moisture in optimal range = no action."""
        sensor = _make_sensor(moisture=55.0)  # In basil range (40-70)
        weather = _make_weather()

        result = self.engine.decide("basil", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.NO_ACTION)
        self.assertEqual(result.water_amount_ml, 0)
        self.assertFalse(result.should_water())

    # ==========================================================
    # Plant-Specific Tests
    # ==========================================================

    def test_pepper_drought_tolerant(self):
        """Pepper has lower min moisture (35%) — drought tolerant."""
        sensor = _make_sensor(moisture=37.0)  # Above pepper min (35)
        weather = _make_weather()

        result = self.engine.decide("pepper", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.NO_ACTION)

    def test_tomato_higher_water_amount(self):
        """Tomato gets 300ml base water."""
        sensor = _make_sensor(moisture=30.0)  # Below tomato min (45)
        weather = _make_weather()

        result = self.engine.decide("tomato", sensor, weather)

        self.assertEqual(result.decision, WaterDecision.WATER_NOW)
        self.assertEqual(result.water_amount_ml, 300)

    # ==========================================================
    # Batch Decisions
    # ==========================================================

    def test_decide_all(self):
        """Test batch decisions for multiple plants."""
        readings = {
            "basil": _make_sensor(moisture=35.0),   # Dry — should water
            "pepper": _make_sensor(moisture=50.0),   # Happy
            "tomato": _make_sensor(moisture=80.0),   # Too wet
        }
        weather = _make_weather()

        results = self.engine.decide_all(readings, weather)

        self.assertEqual(len(results), 3)
        self.assertEqual(results["basil"].decision, WaterDecision.WATER_NOW)
        self.assertEqual(results["pepper"].decision, WaterDecision.NO_ACTION)
        self.assertEqual(results["tomato"].decision, WaterDecision.SKIP_WET)


if __name__ == "__main__":
    unittest.main()
