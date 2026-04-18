"""
julia.sensors.sensor_reader — Read and aggregate sensor data from Home Assistant.

Translates raw HA entity states into Julia's SensorData dataclass.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from julia.core.config import JuliaConfig, SensorMapping
from julia.sensors.ha_client import HomeAssistantClient

logger = logging.getLogger("julia.sensors")


@dataclass
class SensorData:
    """A single sensor reading for one plant."""
    soil_moisture: float       # 0-100%
    temperature: float         # Celsius
    humidity: float            # 0-100%
    sensor_id: str             # Plant ID this reading belongs to
    timestamp: str             # ISO 8601 timestamp
    is_valid: bool = True      # Set to False if any reading failed
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SensorReader:
    """
    Reads sensor data from Home Assistant for all configured plants.

    Uses the sensor_mapping from plants.json to resolve HA entity IDs.
    """

    def __init__(self, config: JuliaConfig, ha_client: HomeAssistantClient):
        self.config = config
        self.ha = ha_client

    def read_plant(self, plant_id: str) -> Optional[SensorData]:
        """
        Read all sensor data for a single plant.

        Returns SensorData or None if the plant has no sensor mapping.
        """
        mapping = self.config.get_sensor_mapping(plant_id)
        if mapping is None:
            logger.warning(f"No sensor mapping for plant '{plant_id}'")
            return None

        warnings = []
        now = datetime.now(timezone.utc).isoformat()

        # Read each sensor value
        moisture = self.ha.get_state_float(mapping.sensor_entity_id)
        temperature = self.ha.get_state_float(mapping.temp_entity_id)
        humidity = self.ha.get_state_float(mapping.humidity_entity_id)

        # Handle missing values with warnings
        is_valid = True

        if moisture is None:
            warnings.append(f"Soil moisture unavailable ({mapping.sensor_entity_id})")
            moisture = -1.0
            is_valid = False

        if temperature is None:
            warnings.append(f"Temperature unavailable ({mapping.temp_entity_id})")
            temperature = 25.0  # Safe default for PR
            # Temperature alone doesn't invalidate the reading

        if humidity is None:
            warnings.append(f"Humidity unavailable ({mapping.humidity_entity_id})")
            humidity = 70.0  # PR average
            # Humidity alone doesn't invalidate the reading

        if warnings:
            for w in warnings:
                logger.warning(w)

        return SensorData(
            soil_moisture=moisture,
            temperature=temperature,
            humidity=humidity,
            sensor_id=plant_id,
            timestamp=now,
            is_valid=is_valid,
            warnings=warnings,
        )

    def read_all(self) -> Dict[str, SensorData]:
        """
        Read sensor data for all plants with sensor mappings.

        Returns a dict of {plant_id: SensorData}.
        """
        results = {}
        for plant_id in self.config.sensor_mappings:
            data = self.read_plant(plant_id)
            if data:
                results[plant_id] = data
                status = "✅" if data.is_valid else "⚠️"
                logger.info(
                    f"{status} {plant_id}: moisture={data.soil_moisture}%, "
                    f"temp={data.temperature}°C, humidity={data.humidity}%"
                )
        return results

    def is_ha_connected(self) -> bool:
        """Check if Home Assistant is reachable."""
        return self.ha.ping()
