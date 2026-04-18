"""
julia.core.brain — Rule-based decision logic for the AI Crop Caretaker.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Decision:
    action: str  # "WATER" or "WAIT"
    reason: str
    model_version: str = "v1-rules"

class Brain:
    """The decision-making engine."""

    def __init__(self):
        self.version = "v1-rules"

    def decide(self, 
               plant_name: str, 
               current_moisture: float, 
               min_moisture: float, 
               rain_probability: float = 0.0) -> Decision:
        """
        Decide whether to water a plant based on its state and weather.
        
        Rules:
        1. If moisture >= min_moisture: WAIT (Hydrated).
        2. If moisture < critical_level (min - 10%): WATER (Emergency).
        3. If moisture < min_moisture AND rain > 70%: WAIT (Rain coming).
        4. Otherwise: WATER.
        """
        
        # Rule 1: Plant is happy
        if current_moisture >= min_moisture:
            return Decision("WAIT", f"Moisture {current_moisture:.1f}% is sufficient (> {min_moisture}%)")

        # Define critical level (way too dry)
        critical_level = max(10.0, min_moisture - 10.0)

        # Rule 2: Emergency watering (Too dry to wait for rain)
        if current_moisture < critical_level:
            return Decision("WATER", f"CRITICAL: Moisture {current_moisture:.1f}% is dangerously low!")

        # Rule 3: Wait for nature
        if rain_probability > 70:
            return Decision("WAIT", f"Rain forecast {rain_probability}% is high. Saving water.")

        # Rule 4: Normal watering
        return Decision("WATER", f"Moisture {current_moisture:.1f}% is below target {min_moisture}%")
