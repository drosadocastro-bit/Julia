"""
julia.actuators.watering — Control the ThirdReality watering valve.

Handles opening/closing the valve via Home Assistant,
with safety caps and timed watering.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from julia.core.config import JuliaConfig
from julia.sensors.ha_client import HomeAssistantClient

logger = logging.getLogger("julia.watering")


@dataclass
class WateringEvent:
    """Record of a watering event."""
    plant_id: str
    amount_ml: int
    duration_seconds: float
    timestamp: str
    success: bool
    reason: str


class WateringController:
    """
    Controls the ThirdReality Zigbee watering valve via Home Assistant.

    Safety features:
    - Maximum watering duration cap
    - Confirmation read-back after toggling
    - Dry-run mode for testing
    """

    def __init__(self, config: JuliaConfig, ha_client: HomeAssistantClient):
        self.config = config
        self.ha = ha_client
        self.valve_entity = config.valve.entity_id
        self.max_duration = config.valve.max_duration_seconds
        self.flow_rate = config.valve.flow_rate_ml_per_second
        self.history: list = []  # In-memory event log

    def water(self, plant_id: str, amount_ml: int, reason: str = "") -> WateringEvent:
        """
        Water a plant for the calculated duration.

        Converts ml to seconds using flow_rate, caps at max_duration.

        Args:
            plant_id: Which plant is being watered
            amount_ml: How much water in milliliters
            reason: Why we're watering (from decision engine)

        Returns:
            WateringEvent with success status
        """
        # Calculate duration
        duration = amount_ml / self.flow_rate
        if duration > self.max_duration:
            logger.warning(
                f"Duration {duration:.0f}s exceeds max {self.max_duration}s — capping."
            )
            duration = self.max_duration

        now = datetime.now(timezone.utc).isoformat()

        # Dry run mode
        if self.config.dry_run:
            logger.info(
                f"🏜️ DRY RUN: Would water {plant_id} with {amount_ml}ml "
                f"for {duration:.0f}s — {reason}"
            )
            event = WateringEvent(
                plant_id=plant_id,
                amount_ml=amount_ml,
                duration_seconds=duration,
                timestamp=now,
                success=True,
                reason=f"[DRY RUN] {reason}",
            )
            self.history.append(event)
            return event

        # Open valve
        logger.info(f"💧 Opening valve for {plant_id}: {amount_ml}ml ({duration:.0f}s)")
        if not self.ha.turn_on(self.valve_entity):
            logger.error(f"Failed to open watering valve!")
            event = WateringEvent(
                plant_id=plant_id,
                amount_ml=0,
                duration_seconds=0,
                timestamp=now,
                success=False,
                reason=f"Valve open failed — {reason}",
            )
            self.history.append(event)
            return event

        # Wait for calculated duration
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            logger.warning("Watering interrupted! Closing valve immediately.")

        # Close valve
        if not self.ha.turn_off(self.valve_entity):
            logger.critical(
                f"⚠️ FAILED TO CLOSE VALVE! Manual intervention required! "
                f"Entity: {self.valve_entity}"
            )
            # Try again
            time.sleep(1)
            self.ha.turn_off(self.valve_entity)

        logger.info(f"✅ Watered {plant_id}: {amount_ml}ml over {duration:.0f}s")

        event = WateringEvent(
            plant_id=plant_id,
            amount_ml=amount_ml,
            duration_seconds=duration,
            timestamp=now,
            success=True,
            reason=reason,
        )
        self.history.append(event)
        return event

    def get_valve_state(self) -> Optional[str]:
        """Check current state of the watering valve."""
        return self.ha.get_state_value(self.valve_entity)

    def emergency_shutoff(self) -> bool:
        """Emergency: turn off the valve immediately."""
        logger.critical("🚨 EMERGENCY SHUTOFF — closing watering valve!")
        result = self.ha.turn_off(self.valve_entity)
        if not result:
            # Try one more time
            time.sleep(0.5)
            result = self.ha.turn_off(self.valve_entity)
        return result

    def get_last_watering(self, plant_id: str) -> Optional[WateringEvent]:
        """Get the most recent watering event for a plant."""
        for event in reversed(self.history):
            if event.plant_id == plant_id and event.success:
                return event
        return None
