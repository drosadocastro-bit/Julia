"""
julia.core.decision_engine — Julia's brain for watering decisions.

Takes sensor data, weather forecasts, and plant health info,
then produces an explainable watering decision.

Philosophy:
- Better to underwater slightly than overwater
- Trust sensors but verify with vision
- Explain every decision
- Learn from outcomes
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional

from julia.core.config import JuliaConfig, PlantProfile
from julia.sensors.sensor_reader import SensorData

logger = logging.getLogger("julia.decision")


# ======================================================================
# Data Types
# ======================================================================

class WaterDecision(Enum):
    """Possible watering decisions."""
    WATER_NOW = "water_now"
    SKIP_RAIN = "skip_rain_forecast"
    SKIP_WET = "skip_soil_wet"
    SKIP_RECENT = "skip_recent_watering"
    SKIP_INVALID = "skip_invalid_sensor"
    EMERGENCY = "emergency_watering"
    NO_ACTION = "no_action_needed"


@dataclass
class WeatherForecast:
    """Weather forecast data."""
    rain_probability_24h: float = 0.0    # 0-100%
    rain_probability_48h: float = 0.0    # 0-100%
    temp_high: float = 30.0             # Celsius
    temp_low: float = 22.0             # Celsius
    humidity: float = 70.0             # 0-100%
    description: str = "unknown"
    is_available: bool = False          # False if weather API failed

    @classmethod
    def unavailable(cls) -> "WeatherForecast":
        """Return a default forecast when weather data is unavailable."""
        return cls(
            rain_probability_24h=0,
            rain_probability_48h=0,
            temp_high=30,
            temp_low=22,
            humidity=70,
            description="Weather data unavailable — using PR defaults",
            is_available=False,
        )


@dataclass
class PlantHealth:
    """Plant health status from vision (Phase 3 — stub for now)."""
    status: str = "unknown"         # healthy, wilting, yellow_leaves, etc.
    confidence: float = 0.0
    details: Optional[str] = None


@dataclass
class WateringResult:
    """The output of a watering decision."""
    decision: WaterDecision
    water_amount_ml: int
    reason: str
    confidence: float
    plant_id: str = ""
    timestamp: str = ""

    def should_water(self) -> bool:
        """Quick check if we should actually water."""
        return self.decision in (WaterDecision.WATER_NOW, WaterDecision.EMERGENCY)


# ======================================================================
# Decision Engine
# ======================================================================

class JuliaDecisionEngine:
    """
    Julia's brain for watering decisions.

    Uses a priority-based rule system:
    1. EMERGENCY — Plant wilting + soil dry → water immediately (1.5x amount)
    2. SKIP_INVALID — Sensor data is invalid → don't water blindly
    3. SKIP_RAIN — Rain forecast > threshold → skip
    4. SKIP_WET — Soil already above max moisture → skip
    5. SKIP_RECENT — Watered too recently → skip
    6. WATER_NOW — Soil below min moisture → water
    7. NO_ACTION — Everything looks good

    The regression model (Phase 4) handles edge cases between
    WATER_NOW and NO_ACTION.
    """

    def __init__(self, config: JuliaConfig, db=None):
        self.config = config
        self._db = db  # Optional JuliaDatabase for persistent cooldown tracking
        self._last_watering: Dict[str, datetime] = {}
        self._decision_callbacks: List[Callable] = []

    def add_decision_callback(self, callback: Callable):
        """
        Register a callback to run after every decision.

        Callback signature: callback(plant_id, result, sensor_data, weather)
        Used by ML collector and database logger.
        """
        self._decision_callbacks.append(callback)

    def _fire_callbacks(self, plant_id: str, result, sensor_data, weather):
        """Fire all registered decision callbacks."""
        for cb in self._decision_callbacks:
            try:
                cb(plant_id, result, sensor_data, weather)
            except Exception as e:
                logger.warning(f"Decision callback failed: {e}")

    def record_watering(self, plant_id: str, timestamp: Optional[datetime] = None):
        """Record when a plant was last watered (for cooldown tracking)."""
        self._last_watering[plant_id] = timestamp or datetime.now(timezone.utc)

    def _hours_since_watering(self, plant_id: str) -> Optional[float]:
        """Get hours since last watering, or None if never watered."""
        # Try in-memory first (most recent)
        last = self._last_watering.get(plant_id)
        if last is not None:
            delta = datetime.now(timezone.utc) - last
            return delta.total_seconds() / 3600

        # Fall back to database if available
        if self._db:
            return self._db.get_hours_since_watering(plant_id)

        return None

    def decide(
        self,
        plant_id: str,
        sensor_data: SensorData,
        weather: WeatherForecast,
        plant_health: Optional[PlantHealth] = None,
        climate_risk: Optional[Dict] = None,
    ) -> WateringResult:
        """
        Main decision function.

        Args:
            plant_id: Key in plants.json (e.g., "basil")
            sensor_data: Current sensor readings
            weather: Weather forecast
            plant_health: Optional vision-based health (Phase 3)
            climate_risk: Optional Multi-Horizon risk evaluation (Phase 12)

        Returns:
            WateringResult with decision, amount, and explanation
        """
        profile = self.config.get_profile(plant_id)
        now = datetime.now(timezone.utc).isoformat()

        # Extract ML risk signals 
        ml_cat = "LOW"
        ml_advisory = ""
        if climate_risk and "composite" in climate_risk:
            ml_cat = climate_risk["composite"].get("category", "LOW")
            ml_advisory = climate_risk["composite"].get("final_advisory", "")

        # ---- Priority 1: EMERGENCY ----
        if plant_health and plant_health.status == "wilting" and plant_health.confidence > 0.6:
            if sensor_data.soil_moisture < profile.min_moisture:
                amount = int(profile.water_amount_ml * 1.5)
                return WateringResult(
                    decision=WaterDecision.EMERGENCY,
                    water_amount_ml=amount,
                    reason=(
                        f"🚨 Emergency: {profile.name} {profile.emoji} is wilting "
                        f"(confidence: {plant_health.confidence:.0%}), "
                        f"soil at {sensor_data.soil_moisture}% (min: {profile.min_moisture}%). "
                        f"Watering {amount}ml immediately."
                    ),
                    confidence=0.95,
                    plant_id=plant_id,
                    timestamp=now,
                )

        # ---- Priority 2: INVALID SENSOR ----
        if not sensor_data.is_valid:
            return WateringResult(
                decision=WaterDecision.SKIP_INVALID,
                water_amount_ml=0,
                reason=(
                    f"⚠️ Sensor data invalid for {profile.name} {profile.emoji}. "
                    f"Warnings: {', '.join(sensor_data.warnings)}. "
                    f"Skipping to avoid blind watering."
                ),
                confidence=0.99,
                plant_id=plant_id,
                timestamp=now,
            )

        # ---- Priority 3: RAIN FORECAST ----
        if (
            weather.is_available
            and weather.rain_probability_24h > self.config.weather.rain_skip_threshold
        ):
            return WateringResult(
                decision=WaterDecision.SKIP_RAIN,
                water_amount_ml=0,
                reason=(
                    f"🌧️ Rain likely ({weather.rain_probability_24h:.0f}% chance in 24h). "
                    f"Skipping for {profile.name} {profile.emoji}."
                ),
                confidence=0.85,
                plant_id=plant_id,
                timestamp=now,
            )

        # ---- Priority 4: SOIL TOO WET ----
        if sensor_data.soil_moisture > profile.max_moisture:
            return WateringResult(
                decision=WaterDecision.SKIP_WET,
                water_amount_ml=0,
                reason=(
                    f"💧 Soil at {sensor_data.soil_moisture}% for {profile.name} {profile.emoji} "
                    f"(max: {profile.max_moisture}%). Already wet enough."
                ),
                confidence=0.90,
                plant_id=plant_id,
                timestamp=now,
            )

        # ---- Priority 5: RECENT WATERING ----
        # If we have a WEEKLY_CRITICAL ML override, we bypass the cooldown to protect the crop
        bypass_cooldown = (ml_advisory == "WEEKLY_CRITICAL_OVERRIDE")
        hours_since = self._hours_since_watering(plant_id)
        
        if not bypass_cooldown and hours_since is not None and hours_since < profile.min_hours_between_watering:
            return WateringResult(
                decision=WaterDecision.SKIP_RECENT,
                water_amount_ml=0,
                reason=(
                    f"⏰ {profile.name} {profile.emoji} was watered {hours_since:.1f}h ago "
                    f"(min: {profile.min_hours_between_watering}h). Too soon."
                ),
                confidence=0.92,
                plant_id=plant_id,
                timestamp=now,
            )

        # ---- Priority 6: SOIL IS DRY — WATER! ----
        if sensor_data.soil_moisture < profile.min_moisture or bypass_cooldown:
            amount = profile.water_amount_ml
            temp_note = ""

            # Adjust for hot days
            if sensor_data.temperature > 30:
                hot_factor = 1.0 + (sensor_data.temperature - 30) * 0.04  # +4% per degree > 30
                hot_factor = min(hot_factor, 1.3)  # Cap at +30%
                amount = int(amount * hot_factor)
                temp_note = f" (adjusted +{(hot_factor - 1) * 100:.0f}% for {sensor_data.temperature}°C)"

            # ML Climate Risk Multiplier
            if ml_cat in ["HIGH", "CRITICAL"]:
                amount = int(amount * 1.3)
                temp_note += f" [ML Risk {ml_cat}: +30% water]"
                
            risk_append = f" ⚠️ ML Advisory: {ml_advisory}" if ml_advisory else ""

            return WateringResult(
                decision=WaterDecision.WATER_NOW if not bypass_cooldown else WaterDecision.EMERGENCY,
                water_amount_ml=amount,
                reason=(
                    f"💧 Soil at {sensor_data.soil_moisture}% for {profile.name} {profile.emoji} "
                    f"(min: {profile.min_moisture}%). Watering {amount}ml{temp_note}.{risk_append}"
                ),
                confidence=0.88,
                plant_id=plant_id,
                timestamp=now,
            )

        # ---- Priority 7: ALL GOOD ----
        risk_append = f" ⚠️ ML Advisory: {ml_advisory}" if ml_advisory else ""
        return WateringResult(
            decision=WaterDecision.NO_ACTION,
            water_amount_ml=0,
            reason=(
                f"✅ {profile.name} {profile.emoji} is happy! "
                f"Soil at {sensor_data.soil_moisture}% "
                f"(range: {profile.min_moisture}-{profile.max_moisture}%).{risk_append}"
            ),
            confidence=0.85,
            plant_id=plant_id,
            timestamp=now,
        )

    def decide_all(
        self,
        sensor_readings: Dict[str, SensorData],
        weather: WeatherForecast,
        plant_health: Optional[Dict[str, PlantHealth]] = None,
    ) -> Dict[str, WateringResult]:
        """Run decisions for all plants at once."""
        results = {}
        for plant_id, data in sensor_readings.items():
            health = plant_health.get(plant_id) if plant_health else None
            result = self.decide(plant_id, data, weather, health)
            results[plant_id] = result
            self._fire_callbacks(plant_id, result, data, weather)
        return results
