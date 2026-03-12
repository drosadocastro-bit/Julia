
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

from julia.data.database import JuliaDatabase


@pytest.fixture
def db(tmp_path):
    db_file = str(tmp_path / "test.db")
    database = JuliaDatabase(db_path=db_file)
    yield database
    database.close()


def test_init_db(db):
    """Tables are created on init."""
    with db._get_conn() as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert {"sensor_readings", "decisions", "weather_snapshots",
            "watering_events", "ml_training_data"}.issubset(tables)


def test_log_sensor_reading(db):
    """Sensor readings are stored and retrievable."""
    db.log_sensor_reading("basil", 45.5, 25.0, 60.0)
    trend = db.get_sensor_trend("basil", hours=1)
    assert len(trend) == 1
    assert trend[0]["soil_moisture"] == 45.5
    assert trend[0]["temperature"] == 25.0


def test_log_decision(db):
    """Decisions are stored and retrievable."""
    db.log_decision(
        plant_id="basil", decision="WATER",
        water_amount_ml=200, reason="Soil is dry",
        confidence=1.0, soil_moisture=35.0,
    )
    history = db.get_decision_history("basil", days=1)
    assert len(history) == 1
    assert history[0]["decision"] == "WATER"
    assert history[0]["reason"] == "Soil is dry"


def test_log_weather(db):
    """Weather snapshots are stored and retrievable."""
    db.log_weather_snapshot(
        temperature=28.0, humidity=75.0,
        rain_24h=20.0, rain_48h=35.0,
        temp_high=31.0, temp_low=25.0,
        description="Partly Cloudy",
    )
    recent = db.get_recent_weather(hours=1)
    assert len(recent) == 1
    assert recent[0]["temperature"] == 28.0
    assert recent[0]["description"] == "Partly Cloudy"


def test_log_watering_event(db):
    """Watering events are stored and retrievable."""
    db.log_watering_event("basil", 200, "Soil dry", decision_type="water_now")
    history = db.get_watering_history("basil", days=1)
    assert len(history) == 1
    assert history[0]["amount_ml"] == 200
