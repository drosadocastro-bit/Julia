"""
julia.core.database — SQLite persistence layer for sensor data and decisions.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _PROJECT_ROOT / "julia.db"

class JuliaDatabase:
    """Handles all interactions with the SQLite database."""

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def _init_db(self):
        """Initialize the database schema if tables don't exist."""
        
        # Sensor Readings Schema
        schema_sensors = """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            plant_id TEXT NOT NULL,
            moisture REAL,
            temperature REAL,
            humidity REAL
        );
        """

        # Decisions Schema
        schema_decisions = """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            plant_id TEXT NOT NULL,
            action TEXT NOT NULL,  -- 'WATER', 'WAIT'
            reason TEXT NOT NULL,
            model_version TEXT DEFAULT 'v1-rules'
        );
        """

        # Weather History Schema
        schema_weather = """
        CREATE TABLE IF NOT EXISTS weather_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            temperature REAL,
            humidity REAL,
            rain_probability REAL,
            is_raining INTEGER DEFAULT 0
        );
        """

        # ML Training Data (Mistakes/Learnings)
        schema_ml = """
        CREATE TABLE IF NOT EXISTS ml_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            plant_id TEXT NOT NULL,
            action TEXT NOT NULL,
            soil_moisture REAL,
            outcome_moisture REAL,
            outcome_health TEXT
        );
        """

        with self._get_conn() as conn:
            conn.execute(schema_sensors)
            conn.execute(schema_decisions)
            conn.execute(schema_weather)
            conn.execute(schema_ml)
            conn.commit()

    def log_sensor_reading(self, plant_id: str, moisture: float, temp: float, humidity: float):
        """Log a single sensor reading."""
        query = """
        INSERT INTO sensor_readings (plant_id, moisture, temperature, humidity)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(query, (plant_id, moisture, temp, humidity))
            conn.commit()

    def log_decision(self, plant_id: str, action: str, reason: str, model_version: str = "v1-rules"):
        """Log a watering decision made by the brain."""
        query = """
        INSERT INTO decisions (plant_id, action, reason, model_version)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(query, (plant_id, action, reason, model_version))
            conn.commit()

    def log_weather_snapshot(self, temp: float, humidity: float, rain_prob: float, is_raining: bool):
        """Log the current weather state."""
        query = """
        INSERT INTO weather_snapshots (temperature, humidity, rain_probability, is_raining)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(query, (temp, humidity, rain_prob, int(is_raining)))
            conn.commit()

    # --- Query Methods (for Dashboard/Analysis) ---

    def get_recent_readings(self, plant_id: str, limit: int = 50) -> List[dict]:
        """Get the most recent sensor readings for a specific plant."""
        query = """
        SELECT * FROM sensor_readings 
        WHERE plant_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        with self._get_conn() as conn:
            cursor = conn.execute(query, (plant_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_decisions(self, limit: int = 20) -> List[dict]:
        """Get the most recent decisions made by Julia."""
        query = """
        SELECT * FROM decisions 
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        with self._get_conn() as conn:
            cursor = conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_sensor_trend(self, plant_id: str, hours: int = 24) -> List[dict]:
        """Get readings for the last N hours for charting."""
        # Note: SQLite content is text, we rely on standard format
        # but for simple retrieval sorting by timestamp works.
        query = """
        SELECT timestamp, moisture, temperature, humidity
        FROM sensor_readings
        WHERE plant_id = ? AND timestamp > datetime('now', ?)
        ORDER BY timestamp ASC
        """
        time_modifier = f"-{hours} hours"
        with self._get_conn() as conn:
            cursor = conn.execute(query, (plant_id, time_modifier))
            return [dict(row) for row in cursor.fetchall()]

    def get_weather_history(self, hours: int = 24) -> List[dict]:
        """Get weather snapshots for the last N hours."""
        query = """
        SELECT timestamp, temperature, humidity, rain_probability, is_raining
        FROM weather_snapshots
        WHERE timestamp > datetime('now', ?)
        ORDER BY timestamp ASC
        """
        time_modifier = f"-{hours} hours"
        with self._get_conn() as conn:
            cursor = conn.execute(query, (time_modifier,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_waterings(self, limit: int = 10) -> List[dict]:
        """Get recent watering events (Decision=WATER)."""
        query = """
        SELECT * FROM decisions 
        WHERE action = 'WATER'
        ORDER BY timestamp DESC 
        LIMIT ?
        """
        with self._get_conn() as conn:
            cursor = conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_bad_outcomes(self, limit: int = 10) -> List[dict]:
        """
        Retrieve decisions where the outcome was negative.
        Currently queries the separate ml_training_data table.
        """
        query = """
        SELECT timestamp, plant_id, action, soil_moisture,
               outcome_moisture, outcome_health
        FROM ml_training_data
        ORDER BY timestamp DESC
        LIMIT ?
        """
        with self._get_conn() as conn:
            cursor = conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def log_training_data(self, plant_id: str, action: str, moisture: float, 
                         outcome_moisture: float, outcome_health: str):
        """Log a decision outcome for learning."""
        query = """
        INSERT INTO ml_training_data 
        (plant_id, action, soil_moisture, outcome_moisture, outcome_health)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            conn.execute(query, (plant_id, action, moisture, outcome_moisture, outcome_health))
            conn.commit()
