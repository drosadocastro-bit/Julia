"""
Tests for julia.sensors.data_validator — Sensor data validation.

Covers:
- Valid readings pass validation
- Out-of-range moisture/temp/humidity caught
- Negative moisture (sensor offline) detected
- Stale timestamps warned about
- PR-specific temperature warnings
- Batch validation
"""

import unittest
from datetime import datetime, timezone, timedelta

from julia.sensors.sensor_reader import SensorData
from julia.sensors.data_validator import DataValidator, ValidationResult


def _make_sensor(
    moisture=50.0, temp=28.0, humidity=70.0,
    valid=True, timestamp=None
):
    """Helper to create test SensorData."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return SensorData(
        soil_moisture=moisture,
        temperature=temp,
        humidity=humidity,
        sensor_id="test_plant",
        timestamp=timestamp,
        is_valid=valid,
    )


class TestDataValidator(unittest.TestCase):
    """Test sensor data validation logic."""

    def setUp(self):
        self.validator = DataValidator()

    # ==========================================================
    # Valid Data
    # ==========================================================

    def test_valid_reading(self):
        """Normal reading passes validation."""
        data = _make_sensor(moisture=55.0, temp=28.0, humidity=72.0)
        result = self.validator.validate(data)

        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors)
        self.assertIsNotNone(result.corrected_data)

    def test_valid_boundary_values(self):
        """Readings at exact boundaries are valid."""
        data = _make_sensor(moisture=0.0, temp=0.0, humidity=100.0)
        result = self.validator.validate(data)

        self.assertTrue(result.is_valid)

    # ==========================================================
    # Moisture Errors
    # ==========================================================

    def test_negative_moisture_error(self):
        """Negative moisture = sensor offline."""
        data = _make_sensor(moisture=-1.0)
        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors)
        self.assertTrue(any("negative" in e.lower() for e in result.errors))

    def test_moisture_above_100_error(self):
        """Moisture > 100% = sensor malfunction."""
        data = _make_sensor(moisture=105.0)
        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)
        self.assertTrue(any("malfunction" in e.lower() for e in result.errors))

    # ==========================================================
    # Temperature Checks
    # ==========================================================

    def test_extreme_high_temp_error(self):
        """Temperature above 55°C = sensor malfunction."""
        data = _make_sensor(temp=60.0)
        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)

    def test_low_temp_warning(self):
        """Temperature unusually low for PR = warning."""
        data = _make_sensor(temp=-5.0)
        result = self.validator.validate(data)

        # Low temp is a warning, not an error
        self.assertTrue(result.has_warnings)

    def test_cool_temp_pr_warning(self):
        """Temperature 0-10°C is rare in PR — warns."""
        data = _make_sensor(temp=8.0)
        result = self.validator.validate(data)

        self.assertTrue(result.has_warnings)
        self.assertTrue(any("Puerto Rico" in w for w in result.warnings))

    # ==========================================================
    # Humidity Checks
    # ==========================================================

    def test_humidity_out_of_range_warning(self):
        """Humidity outside 0-100% = warning."""
        data = _make_sensor(humidity=110.0)
        result = self.validator.validate(data)

        self.assertTrue(result.has_warnings)

    # ==========================================================
    # Staleness
    # ==========================================================

    def test_stale_data_warning(self):
        """Reading older than 2 hours = stale warning."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        data = _make_sensor(timestamp=old_time)
        result = self.validator.validate(data)

        self.assertTrue(result.has_warnings)
        self.assertTrue(any("stale" in w.lower() for w in result.warnings))

    def test_fresh_data_no_stale_warning(self):
        """Recent reading = no stale warning."""
        data = _make_sensor()  # Default is now
        result = self.validator.validate(data)

        self.assertFalse(any("stale" in w.lower() for w in result.warnings))

    # ==========================================================
    # Original Validity Flag
    # ==========================================================

    def test_invalid_flag_from_reader(self):
        """If sensor_reader flagged as invalid, validation should error."""
        data = _make_sensor(valid=False)
        result = self.validator.validate(data)

        self.assertFalse(result.is_valid)

    # ==========================================================
    # Batch Validation
    # ==========================================================

    def test_batch_validation(self):
        """Validate multiple readings at once."""
        readings = {
            "basil": _make_sensor(moisture=55.0),     # Valid
            "pepper": _make_sensor(moisture=-1.0),    # Invalid
            "tomato": _make_sensor(moisture=50.0, temp=8.0),  # Valid with warning
        }

        results = self.validator.validate_batch(readings)

        self.assertEqual(len(results), 3)
        self.assertTrue(results["basil"].is_valid)
        self.assertFalse(results["pepper"].is_valid)
        self.assertTrue(results["tomato"].is_valid)  # Warning only, still valid
        self.assertTrue(results["tomato"].has_warnings)


if __name__ == "__main__":
    unittest.main()
