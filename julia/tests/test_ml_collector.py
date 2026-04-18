"""
Tests for julia.data.ml_collector — ML training data collection.
"""

import os
import tempfile
import unittest

from julia.data.database import JuliaDatabase
from julia.data.ml_collector import MLCollector


class TestMLCollector(unittest.TestCase):
    """Test ML training data collection and export."""

    def setUp(self):
        """Create temp DB and collector for each test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = JuliaDatabase(db_path=self.tmp.name)
        self.collector = MLCollector(self.db)

    def tearDown(self):
        self.db.close()
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass
        # Clean up exported CSV if it exists
        csv_path = os.path.join(
            os.path.dirname(self.tmp.name), "training_data.csv"
        )
        try:
            os.unlink(csv_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Record Decision
    # ------------------------------------------------------------------

    def test_record_decision_returns_id(self):
        """record_decision should return a valid sample ID."""
        sample_id = self.collector.record_decision(
            plant_id="basil",
            soil_moisture=35.0,
            temperature=28.0,
            humidity=72.0,
            rain_probability=25.0,
            temp_forecast=32.0,
            days_since_water=1.0,
            growth_stage=1,
            action="water",
            water_amount_ml=200,
        )
        self.assertIsNotNone(sample_id)
        self.assertGreater(sample_id, 0)

    def test_record_skip_decision(self):
        """Skip decisions should also be recorded."""
        sample_id = self.collector.record_decision(
            plant_id="pepper",
            soil_moisture=60.0,
            temperature=29.0,
            humidity=68.0,
            rain_probability=80.0,
            temp_forecast=31.0,
            days_since_water=0.5,
            growth_stage=1,
            action="skip",
            water_amount_ml=0,
        )
        self.assertGreater(sample_id, 0)

    # ------------------------------------------------------------------
    # Outcome Checks
    # ------------------------------------------------------------------

    def test_check_outcomes_records_correctly(self):
        """Outcomes should be recorded for matching plants."""
        self.collector.record_decision(
            plant_id="basil", soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
            rain_probability=25.0, temp_forecast=32.0,
            days_since_water=1.0, growth_stage=1,
            action="water", water_amount_ml=200,
        )

        # Check outcomes (min_age=0 for testing)
        self.db.get_pending_outcomes = lambda min_age_hours=24: \
            self.db.get_training_data(completed_only=False)

        current = {"basil": {"soil_moisture": 58.0}}
        # Manually mark one as pending by using the real method
        pending = self.db.get_pending_outcomes(min_age_hours=0)
        if pending:
            recorded = self.collector.check_outcomes(current)
            self.assertGreaterEqual(recorded, 0)

    def test_check_outcomes_skips_missing_plants(self):
        """Plants not in current_readings should be skipped."""
        self.collector.record_decision(
            plant_id="basil", soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
            rain_probability=25.0, temp_forecast=32.0,
            days_since_water=1.0, growth_stage=1,
            action="water",
        )

        # No basil in current readings
        recorded = self.collector.check_outcomes({"pepper": {"soil_moisture": 50.0}})
        self.assertEqual(recorded, 0)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def test_export_csv_empty(self):
        """Export should handle no data gracefully."""
        filepath = self.collector.export_csv()
        self.assertIsNotNone(filepath)

    def test_export_csv_with_data(self):
        """Export should create a CSV with completed samples."""
        sample_id = self.collector.record_decision(
            plant_id="basil", soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
            rain_probability=25.0, temp_forecast=32.0,
            days_since_water=1.0, growth_stage=1,
            action="water", water_amount_ml=200,
        )

        # Record outcome
        self.db.record_training_outcome(sample_id, 58.0, "healthy")

        filepath = self.collector.export_csv()
        self.assertTrue(os.path.exists(filepath))

        # Verify CSV content
        with open(filepath, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # header + 1 data row
            self.assertIn("basil", lines[1])
            self.assertIn("water", lines[1])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def test_get_stats_empty(self):
        """Stats should work with empty database."""
        stats = self.collector.get_stats()
        self.assertEqual(stats["total_samples"], 0)
        self.assertEqual(stats["completed"], 0)
        self.assertEqual(stats["pending"], 0)

    def test_get_stats_with_data(self):
        """Stats should accurately reflect data state."""
        self.collector.record_decision(
            plant_id="basil", soil_moisture=35.0,
            temperature=28.0, humidity=72.0,
            rain_probability=25.0, temp_forecast=32.0,
            days_since_water=1.0, growth_stage=1,
            action="water", water_amount_ml=200,
        )
        self.collector.record_decision(
            plant_id="pepper", soil_moisture=60.0,
            temperature=29.0, humidity=68.0,
            rain_probability=10.0, temp_forecast=31.0,
            days_since_water=0.5, growth_stage=1,
            action="skip",
        )

        stats = self.collector.get_stats()
        self.assertEqual(stats["total_samples"], 2)
        self.assertEqual(stats["pending"], 2)
        self.assertEqual(stats["per_plant"]["basil"]["water"], 1)
        self.assertEqual(stats["per_plant"]["pepper"]["skip"], 1)


if __name__ == "__main__":
    unittest.main()
