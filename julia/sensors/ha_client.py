"""
julia.sensors.ha_client — Home Assistant REST API client.

Provides methods to read sensor states and call services
(like toggling the watering valve) through HA's REST API.
"""

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("julia.ha_client")


class HomeAssistantError(Exception):
    """Raised when Home Assistant API returns an error."""
    pass


class HomeAssistantClient:
    """
    Client for the Home Assistant REST API.

    Requires a long-lived access token generated from:
    HA → Profile → Long-Lived Access Tokens → Create Token
    """

    def __init__(self, url: str, token: str, timeout: int = 10):
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if not token:
            logger.warning("No HA token configured — API calls will fail.")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check if Home Assistant is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/",
                headers=self._headers,
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.debug(f"HA ping failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Entity States
    # ------------------------------------------------------------------

    def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of an entity.

        Returns the full state dict or None if unreachable.
        Example return:
        {
            "entity_id": "sensor.basil_soil_moisture",
            "state": "42.5",
            "attributes": {"unit_of_measurement": "%", "friendly_name": "Basil Soil Moisture"},
            "last_changed": "2026-02-17T14:30:00+00:00",
            "last_updated": "2026-02-17T14:30:00+00:00"
        }
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/states/{entity_id}",
                headers=self._headers,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                logger.warning(f"Entity not found: {entity_id}")
                return None
            else:
                logger.error(f"HA returned {resp.status_code} for {entity_id}")
                return None
        except requests.RequestException as e:
            logger.error(f"Failed to get state for {entity_id}: {e}")
            return None

    def get_state_value(self, entity_id: str) -> Optional[str]:
        """Get just the state value string for an entity."""
        state = self.get_state(entity_id)
        if state:
            return state.get("state")
        return None

    def get_state_float(self, entity_id: str) -> Optional[float]:
        """Get the state value as a float, or None if unavailable/invalid."""
        value = self.get_state_value(entity_id)
        if value is None or value in ("unavailable", "unknown"):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot parse '{value}' as float for {entity_id}")
            return None

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------

    def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Call a Home Assistant service.

        Examples:
            call_service("switch", "turn_on", entity_id="switch.watering_valve")
            call_service("switch", "turn_off", entity_id="switch.watering_valve")

        Returns True on success, False on failure.
        """
        payload = data or {}
        if entity_id:
            payload["entity_id"] = entity_id

        try:
            resp = requests.post(
                f"{self.base_url}/api/services/{domain}/{service}",
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                logger.info(f"Service called: {domain}.{service} on {entity_id}")
                return True
            else:
                logger.error(f"Service call failed ({resp.status_code}): {resp.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Service call failed: {e}")
            return False

    def turn_on(self, entity_id: str) -> bool:
        """Turn on a switch/light entity."""
        domain = entity_id.split(".")[0]
        return self.call_service(domain, "turn_on", entity_id=entity_id)

    def turn_off(self, entity_id: str) -> bool:
        """Turn off a switch/light entity."""
        domain = entity_id.split(".")[0]
        return self.call_service(domain, "turn_off", entity_id=entity_id)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def get_all_states(self) -> list:
        """Get all entity states (useful for debugging)."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/states",
                headers=self._headers,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except requests.RequestException as e:
            logger.error(f"Failed to get all states: {e}")
            return []

    def __repr__(self) -> str:
        return f"HomeAssistantClient(url={self.base_url})"
