"""
Tests for julia.data.database — SQLite persistence layer.
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from julia.data.database import JuliaDatabase


class TestJuliaDatabase(unittest.TestCase):
    """Test the SQLite database layer."""

    def setUp(self):
        """Create a temp database for each test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = JuliaDatabase(db_path=self.tmp.name)

    def tearDown(self):
        self.db.close()
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Sensor Readings
    # ------------------------------------------------------------------

    def test_log_and_query_sensor_readings(self):
        """Sensor readings can be logged and retrieved."""
        self.db.log_sensor_reading("basil", 55.0, 28.0, 72.0)
        self.db.log_sensor_reading("basil", 52.0, 29.0, 70.0)

        trend = self.db.get_sensor_trend("basil", hours=1)
        self.assertEqual(len(trend), 2)
        self.assertEqual(trend[0]["soil_moisture"], 55.0)

    def test_sensor_warnings_stored_as_json(self):
        """Warnings should be stored as JSON array."""
        self.db.log_sensor_reading(
            "pepper", 45.0, 30.0, 65.0,
            warnings=["Humidity low", "Temp high"]
        )
        trend = self.db.get_sensor_trend("pepper", hours=1)
        self.assertEqual(len(trend), 1)

    # ------------------------------------------------------------------
    # Watering Events
    # ------------------------------------------------------------------

    def test_log_watering_event(self):
        """Watering events can be logged and retrieved."""
        self.db.log_watering_event(
            plant_id="basil", amount_ml=200,
            reason="Soil dry", decision_type="water_now",
            duration_seconds=40.0, moisture_before=35.0,
        )

        history = self.db.get_watering_history("basil", days=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["amount_ml"], 200)

    def test_get_last_watering(self):
        """Should return the most recent watering event."""
        self.db.log_watering_event("basil", 100, "Test 1")
        self.db.log_watering_event("basil", 200, "Test 2")

        last = self.db.get_last_watering("basil")
        self.assertIsNotNone(last)
        self.assertEqual(last["amount_ml"], 200)

    def test_get_last_watering_none(self):
        """Should return None for an unwatered plant."""
        last = self.db.get_last_watering("cilantro")
        self.assertIsNone(last)

    def test_get_watering_history_all_plants(self):
        """History without plant_id returns all plants."""
        self.db.log_watering_event("basil", 200, "Dry")
        self.db.log_watering_event("pepper", 250, "Dry")

        history = self.db.get_watering_history(days=1)
        self.assertEqual(len(history), 2)

    def test_unchecked_waterings(self):
        """unchecked waterings are those without outcome checks."""
        self.db.log_watering_event("basil", 200, "Test")
        # min_age_hours=0 so it shows up immediately
        unchecked = self.db.get_unchecked_waterings(min_age_hours=0)
        self.assertEqual(len(unchecked), 1)

    def test_mark_outcome_checked(self):
        """Marking outcome should make it disappear from unchecked."""
        self.db.log_watering_event("basil", 200, "Test")
        unchecked = self.db.get_unchecked_waterings(min_age_hours=0)
        self.db.mark_outcome_checked(unchecked[0]["id"])
        unchecked2 = self.db.get_unchecked_waterings(min_age_hours=0)
        self.assertEqual(len(unchecked2), 0)

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def test_log_decision(self):
        """Decisions can be logged and retrieved."""
        self.db.log_decision(
            plant_id="basil", decision="water_now",
            water_amount_ml=200, reason="Soil dry",
            confidence=0.88, soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
        )

        history = self.db.get_decision_history("basil", days=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["decision"], "water_now")

    # ------------------------------------------------------------------
    # Weather Snapshots
    # ------------------------------------------------------------------

    def test_log_weather_snapshot(self):
        """Weather snapshots can be logged and retrieved."""
        self.db.log_weather_snapshot(
            temperature=28.0, humidity=72.0,
            rain_24h=30.0, rain_48h=45.0,
            temp_high=32.0, temp_low=24.0,
            description="Partly cloudy",
        )

        weather = self.db.get_recent_weather(hours=1)
        self.assertEqual(len(weather), 1)
        self.assertEqual(weather[0]["description"], "Partly cloudy")

    # ------------------------------------------------------------------
    # ML Training Data
    # ------------------------------------------------------------------

    def test_log_training_sample(self):
        """Training samples can be logged with features and action."""
        sample_id = self.db.log_training_sample(
            plant_id="basil", soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
            rain_probability=25.0, temp_forecast=32.0,
            days_since_water=1.0, growth_stage=1,
            action="water", water_amount_ml=200,
        )
        self.assertIsNotNone(sample_id)
        self.assertGreater(sample_id, 0)

    def test_record_outcome(self):
        """24h outcomes can be recorded for training samples."""
        sample_id = self.db.log_training_sample(
            plant_id="basil", soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
            rain_probability=25.0, temp_forecast=32.0,
            days_since_water=1.0, growth_stage=1,
            action="water", water_amount_ml=200,
        )

        self.db.record_training_outcome(sample_id, 58.0, "healthy")

        data = self.db.get_training_data(completed_only=True)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["outcome_moisture"], 58.0)

    def test_get_pending_outcomes(self):
        """Pending outcomes are those not yet recorded."""
        self.db.log_training_sample(
            plant_id="basil", soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
            rain_probability=25.0, temp_forecast=32.0,
            days_since_water=1.0, growth_stage=1,
            action="water",
        )
        # With 0 hours min age, should show up
        pending = self.db.get_pending_outcomes(min_age_hours=0)
        self.assertEqual(len(pending), 1)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def test_daily_stats(self):
        """Daily stats aggregate watering events."""
        self.db.log_watering_event("basil", 200, "Dry")
        self.db.log_watering_event("pepper", 250, "Dry")

        stats = self.db.get_daily_stats(days=1)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["total_waterings"], 2)
        self.assertEqual(stats[0]["total_water_ml"], 450)

    def test_plant_summary(self):
        """Plant summary aggregates lifetime stats."""
        self.db.log_watering_event("basil", 200, "Test")
        self.db.log_sensor_reading("basil", 55.0, 28.0, 72.0)

        summary = self.db.get_plant_summary("basil")
        self.assertEqual(summary["plant_id"], "basil")
        self.assertEqual(summary["watering"]["total_waterings"], 1)
        self.assertEqual(summary["latest_sensor"]["soil_moisture"], 55.0)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def test_prune_old_data(self):
        """Pruning should not affect recent data."""
        self.db.log_sensor_reading("basil", 55.0, 28.0, 72.0)
        self.db.prune_old_data()

        trend = self.db.get_sensor_trend("basil", hours=1)
        self.assertEqual(len(trend), 1)  # Still there


if __name__ == "__main__":
    unittest.main()
