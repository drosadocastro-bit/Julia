"""
julia.core.config — Configuration management for Julia.

Loads settings from environment variables and plants.json.
Provides typed dataclass configs used by all other modules.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

logger = logging.getLogger("julia.config")

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class HomeAssistantConfig:
    """Home Assistant connection settings."""
    url: str = "http://homeassistant.local:8123"
    token: str = ""
    timeout: int = 10  # seconds


@dataclass
class WeatherConfig:
    """OpenWeather API settings."""
    api_key: str = ""
    latitude: float = 18.2208  # Puerto Rico default
    longitude: float = -66.5901
    location_name: str = "Puerto Rico"
    rain_skip_threshold: float = 60.0  # % chance to skip watering
    cache_ttl_minutes: int = 30


@dataclass
class WateringValveConfig:
    """Watering valve settings."""
    entity_id: str = "switch.watering_valve"
    max_duration_seconds: int = 300  # Safety cap: 5 minutes max
    flow_rate_ml_per_second: float = 5.0  # Calibrate with your setup


@dataclass
class NotificationConfig:
    """Notification backend settings."""
    pushover_user_key: str = ""
    pushover_api_token: str = ""
    enable_pushover: bool = False
    enable_console: bool = True
    enable_file_log: bool = True


@dataclass
class ScheduleConfig:
    """Scheduling settings."""
    check_interval_hours: int = 6
    vision_check_interval_hours: int = 12
    sensor_read_interval_minutes: int = 15


@dataclass
class PlantProfile:
    """A single plant's care profile."""
    name: str
    emoji: str = "🌱"
    min_moisture: float = 40.0
    max_moisture: float = 70.0
    optimal_moisture: float = 55.0
    water_amount_ml: int = 200
    min_hours_between_watering: int = 12
    drought_tolerant: bool = False
    growth_stages: list = field(default_factory=lambda: ["seedling", "growing", "mature"])
    notes: str = ""


@dataclass
class SensorMapping:
    """Maps a plant to its Home Assistant sensor entity IDs."""
    sensor_entity_id: str
    temp_entity_id: str
    humidity_entity_id: str


class JuliaConfig:
    """
    Central configuration for the Julia system.

    Loads from environment variables and plants.json.
    All modules should use this single config object.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Resolve config path
        if config_path is None:
            config_path = os.getenv(
                "JULIA_PLANTS_CONFIG",
                str(_PROJECT_ROOT / "julia" / "data" / "plants.json")
            )
        self.config_path = Path(config_path)

        # General settings
        self.dry_run: bool = os.getenv("JULIA_DRY_RUN", "false").lower() == "true"
        self.log_level: str = os.getenv("JULIA_LOG_LEVEL", "INFO")

        # Sub-configs from env
        self.home_assistant = HomeAssistantConfig(
            url=os.getenv("HA_URL", "http://homeassistant.local:8123"),
            token=os.getenv("HA_TOKEN", ""),
        )
        self.weather = WeatherConfig(
            api_key=os.getenv("OPENWEATHER_API_KEY", ""),
        )
        self.notifications = NotificationConfig(
            pushover_user_key=os.getenv("PUSHOVER_USER_KEY", ""),
            pushover_api_token=os.getenv("PUSHOVER_API_TOKEN", ""),
        )
        self.notifications.enable_pushover = bool(
            self.notifications.pushover_user_key and self.notifications.pushover_api_token
        )
        self.valve = WateringValveConfig()
        self.schedule = ScheduleConfig()

        # Plant profiles and sensor mappings from JSON
        self.plant_profiles: Dict[str, PlantProfile] = {}
        self.sensor_mappings: Dict[str, SensorMapping] = {}
        self._load_config_file()

    def _load_config_file(self):
        """Load plant profiles and sensor mappings from plants.json."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self._load_defaults()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
            self._load_defaults()
            return

        # Load plant profiles
        for plant_id, profile_data in data.get("plants", {}).items():
            self.plant_profiles[plant_id] = PlantProfile(
                name=profile_data.get("name", plant_id.capitalize()),
                emoji=profile_data.get("emoji", "🌱"),
                min_moisture=profile_data.get("min_moisture", 40),
                max_moisture=profile_data.get("max_moisture", 70),
                optimal_moisture=profile_data.get("optimal_moisture", 55),
                water_amount_ml=profile_data.get("water_amount_ml", 200),
                min_hours_between_watering=profile_data.get("min_hours_between_watering", 12),
                drought_tolerant=profile_data.get("drought_tolerant", False),
                growth_stages=profile_data.get("growth_stages", ["seedling", "growing", "mature"]),
                notes=profile_data.get("notes", ""),
            )

        # Load sensor mappings
        for plant_id, mapping_data in data.get("sensor_mapping", {}).items():
            self.sensor_mappings[plant_id] = SensorMapping(
                sensor_entity_id=mapping_data["sensor_entity_id"],
                temp_entity_id=mapping_data["temp_entity_id"],
                humidity_entity_id=mapping_data["humidity_entity_id"],
            )

        # Load valve config
        valve_data = data.get("watering_valve", {})
        if valve_data:
            self.valve = WateringValveConfig(
                entity_id=valve_data.get("entity_id", self.valve.entity_id),
                max_duration_seconds=valve_data.get("max_duration_seconds", self.valve.max_duration_seconds),
                flow_rate_ml_per_second=valve_data.get("flow_rate_ml_per_second", self.valve.flow_rate_ml_per_second),
            )

        # Load weather config from JSON (API key still from env)
        weather_data = data.get("weather", {})
        if weather_data:
            self.weather.latitude = weather_data.get("latitude", self.weather.latitude)
            self.weather.longitude = weather_data.get("longitude", self.weather.longitude)
            self.weather.location_name = weather_data.get("location_name", self.weather.location_name)
            self.weather.rain_skip_threshold = weather_data.get("rain_skip_threshold", self.weather.rain_skip_threshold)

        # Load schedule config
        schedule_data = data.get("schedule", {})
        if schedule_data:
            self.schedule = ScheduleConfig(
                check_interval_hours=schedule_data.get("check_interval_hours", 6),
                vision_check_interval_hours=schedule_data.get("vision_check_interval_hours", 12),
                sensor_read_interval_minutes=schedule_data.get("sensor_read_interval_minutes", 15),
            )

        logger.info(f"Loaded {len(self.plant_profiles)} plant profiles, {len(self.sensor_mappings)} sensor mappings")

    def _load_defaults(self):
        """Load hardcoded default plant profiles."""
        self.plant_profiles = {
            "basil": PlantProfile(name="Basil", emoji="🌿", min_moisture=40, max_moisture=70,
                                  optimal_moisture=55, water_amount_ml=200),
        }

    def get_profile(self, plant_id: str) -> PlantProfile:
        """Get plant profile by ID, falling back to basil defaults."""
        return self.plant_profiles.get(plant_id, self.plant_profiles.get("basil", PlantProfile(name="Unknown")))

    def get_sensor_mapping(self, plant_id: str) -> Optional[SensorMapping]:
        """Get sensor mapping for a plant."""
        return self.sensor_mappings.get(plant_id)

    def __repr__(self) -> str:
        return (
            f"JuliaConfig(\n"
            f"  plants={list(self.plant_profiles.keys())},\n"
            f"  sensors={list(self.sensor_mappings.keys())},\n"
            f"  ha_url={self.home_assistant.url},\n"
            f"  dry_run={self.dry_run},\n"
            f"  log_level={self.log_level}\n"
            f")"
        )
