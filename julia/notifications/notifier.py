"""
julia.notifications.notifier — Alert and notification system.

Supports multiple backends:
- Console (always on)
- File log (always on)
- Pushover (optional, cloud-based push)
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from julia.core.config import JuliaConfig

logger = logging.getLogger("julia.notify")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Notifier:
    """
    Multi-backend notification system for Julia.

    Sends alerts through configured backends based on severity.
    Critical alerts are sent to ALL backends.
    """

    # Emoji map for alert levels
    LEVEL_EMOJI = {
        AlertLevel.INFO: "ℹ️",
        AlertLevel.WARNING: "⚠️",
        AlertLevel.CRITICAL: "🚨",
    }

    def __init__(self, config: JuliaConfig):
        self.config = config
        self._log_file = Path(config.config_path).parent / "julia_alerts.jsonl"
        self._pushover_available = False

        # Try to import Pushover if configured
        if config.notifications.enable_pushover:
            try:
                import httpx  # or requests — we'll use requests
                self._pushover_available = True
            except ImportError:
                logger.warning("Pushover configured but 'requests' not available.")

    def send(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        title: Optional[str] = None,
    ):
        """
        Send a notification through all configured backends.

        Args:
            message: The alert message
            level: Severity level
            title: Optional title (defaults to "Julia 🌱")
        """
        title = title or "Julia 🌱"
        emoji = self.LEVEL_EMOJI.get(level, "")
        full_message = f"{emoji} {message}"

        # Console — always
        if self.config.notifications.enable_console:
            self._send_console(title, full_message, level)

        # File log — always
        if self.config.notifications.enable_file_log:
            self._send_file_log(title, message, level)

        # Pushover — only if configured and for warnings/critical
        if (
            self._pushover_available
            and self.config.notifications.enable_pushover
            and level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
        ):
            self._send_pushover(title, message, level)

    def info(self, message: str, title: Optional[str] = None):
        """Send an INFO-level notification."""
        self.send(message, AlertLevel.INFO, title)

    def warning(self, message: str, title: Optional[str] = None):
        """Send a WARNING-level notification."""
        self.send(message, AlertLevel.WARNING, title)

    def critical(self, message: str, title: Optional[str] = None):
        """Send a CRITICAL-level notification."""
        self.send(message, AlertLevel.CRITICAL, title)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _send_console(self, title: str, message: str, level: AlertLevel):
        """Print to console with color-coded level."""
        if level == AlertLevel.CRITICAL:
            logger.critical(f"[{title}] {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"[{title}] {message}")
        else:
            logger.info(f"[{title}] {message}")

    def _send_file_log(self, title: str, message: str, level: AlertLevel):
        """Append to JSONL log file."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level.value,
                "title": title,
                "message": message,
            }
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.debug(f"Failed to write alert log: {e}")

    def _send_pushover(self, title: str, message: str, level: AlertLevel):
        """Send push notification via Pushover API."""
        import requests

        priority_map = {
            AlertLevel.INFO: -1,       # Low priority
            AlertLevel.WARNING: 0,     # Normal
            AlertLevel.CRITICAL: 1,    # High priority
        }

        try:
            resp = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": self.config.notifications.pushover_api_token,
                    "user": self.config.notifications.pushover_user_key,
                    "title": title,
                    "message": message,
                    "priority": priority_map.get(level, 0),
                },
                timeout=10,
            )
            if resp.status_code == 200:
                logger.debug(f"Pushover notification sent: {title}")
            else:
                logger.warning(f"Pushover failed ({resp.status_code}): {resp.text}")
        except requests.RequestException as e:
            logger.warning(f"Pushover send failed: {e}")

    # ------------------------------------------------------------------
    # Decision Notifications
    # ------------------------------------------------------------------

    def notify_decision(self, plant_id: str, decision_reason: str, should_water: bool):
        """Send a notification about a watering decision."""
        if should_water:
            self.send(decision_reason, AlertLevel.INFO, f"Julia — Watering {plant_id}")
        else:
            # Only log skips, don't push-notify for them
            self._send_console(f"Julia — {plant_id}", decision_reason, AlertLevel.INFO)
            self._send_file_log(f"Julia — {plant_id}", decision_reason, AlertLevel.INFO)

    def notify_emergency(self, plant_id: str, reason: str):
        """Send a CRITICAL notification for emergency watering."""
        self.critical(reason, f"Julia — EMERGENCY: {plant_id}")

    def notify_sensor_failure(self, plant_id: str, errors: list):
        """Notify about sensor failures."""
        error_text = "; ".join(errors)
        self.warning(
            f"Sensor issues for {plant_id}: {error_text}",
            "Julia — Sensor Alert"
        )
