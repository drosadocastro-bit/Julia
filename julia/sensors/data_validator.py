"""
julia.sensors.data_validator — Validate sensor readings.

Catches bad sensor data before it reaches the decision engine:
- Out-of-range values (sensor malfunction)
- Stale timestamps (sensor offline)
- Impossible readings (moisture > 100%, negative temps in PR, etc.)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from julia.sensors.sensor_reader import SensorData

logger = logging.getLogger("julia.validator")


@dataclass
class ValidationResult:
    """Result of validating a sensor reading."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    corrected_data: Optional[SensorData] = None

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class DataValidator:
    """
    Validates sensor readings for sanity.

    Catches hardware failures, stale data, and impossible values.
    """

    # Sane ranges for sensor values
    MOISTURE_MIN = 0.0
    MOISTURE_MAX = 100.0
    TEMP_MIN = -10.0        # Celsius — extremely unlikely in PR
    TEMP_MAX = 55.0         # Above this, sensor is probably broken
    HUMIDITY_MIN = 0.0
    HUMIDITY_MAX = 100.0
    STALE_THRESHOLD_HOURS = 2  # Reading older than this is stale

    def validate(self, data: SensorData) -> ValidationResult:
        """
        Validate a single sensor reading.

        Returns a ValidationResult with warnings and/or errors.
        If there are only warnings, the data is still usable.
        If there are errors, the data should NOT be used for decisions.
        """
        warnings = []
        errors = []

        # ---- Moisture ----
        if data.soil_moisture < 0:
            errors.append(
                f"Soil moisture is negative ({data.soil_moisture}%) — sensor likely offline"
            )
        elif data.soil_moisture < self.MOISTURE_MIN:
            errors.append(
                f"Soil moisture below range: {data.soil_moisture}%"
            )
        elif data.soil_moisture > self.MOISTURE_MAX:
            errors.append(
                f"Soil moisture above 100%: {data.soil_moisture}% — sensor malfunction"
            )

        # ---- Temperature ----
        if data.temperature < self.TEMP_MIN:
            warnings.append(
                f"Temperature unusually low: {data.temperature}°C"
            )
        elif data.temperature > self.TEMP_MAX:
            errors.append(
                f"Temperature above {self.TEMP_MAX}°C: {data.temperature}°C — sensor malfunction?"
            )

        # PR-specific: temps below 10°C are extremely rare
        if self.TEMP_MIN <= data.temperature < 10:
            warnings.append(
                f"Temperature very low for Puerto Rico: {data.temperature}°C — double check sensor"
            )

        # ---- Humidity ----
        if data.humidity < self.HUMIDITY_MIN or data.humidity > self.HUMIDITY_MAX:
            warnings.append(
                f"Humidity out of range: {data.humidity}%"
            )

        # ---- Staleness ----
        try:
            reading_time = datetime.fromisoformat(data.timestamp)
            # Ensure timezone awareness
            if reading_time.tzinfo is None:
                reading_time = reading_time.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - reading_time
            if age > timedelta(hours=self.STALE_THRESHOLD_HOURS):
                warnings.append(
                    f"Reading is {age.total_seconds() / 3600:.1f}h old — data may be stale"
                )
        except (ValueError, TypeError):
            warnings.append("Could not parse timestamp for freshness check")

        # ---- Original validity ----
        if not data.is_valid:
            errors.append("Sensor reader flagged this reading as invalid")

        # Build result
        is_valid = len(errors) == 0

        if warnings:
            for w in warnings:
                logger.warning(f"[{data.sensor_id}] {w}")
        if errors:
            for e in errors:
                logger.error(f"[{data.sensor_id}] {e}")

        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            corrected_data=data if is_valid else None,
        )

    def validate_batch(self, readings: dict) -> dict:
        """
        Validate a batch of sensor readings.

        Args:
            readings: Dict of {plant_id: SensorData}

        Returns:
            Dict of {plant_id: ValidationResult}
        """
        results = {}
        for plant_id, data in readings.items():
            results[plant_id] = self.validate(data)
        return results
