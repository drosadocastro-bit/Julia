"""
julia.core.weather — Real-world weather service using Open-Meteo API with simulation fallback.
"""

import requests
import random
from typing import Dict, Any, Optional
from datetime import datetime

# Open-Meteo API config (San Juan, PR default)
_LAT = 18.4655
_LON = -66.1057
_API_URL = "https://api.open-meteo.com/v1/forecast"

class WeatherService:
    """Fetches real weather or generates fallback data."""

    def __init__(self, lat: float = _LAT, lon: float = _LON):
        self.lat = lat
        self.lon = lon
        self.current_weather = self._generate_fallback()

    def update(self) -> Dict[str, Any]:
        """Fetch latest weather, falling back to simulation on error."""
        try:
            data = self._fetch_real_weather()
            if data:
                self.current_weather = data
                return data
        except Exception as e:
            print(f"Weather API Error: {e}")
        
        # Fallback if API fails
        self.current_weather = self._generate_fallback(self.current_weather)
        return self.current_weather

    def _fetch_real_weather(self) -> Optional[Dict[str, Any]]:
        """Query Open-Meteo for current conditions."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
            "daily": "precipitation_probability_max",
            "timezone": "auto"
        }
        
        resp = requests.get(_API_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        curr = data.get("current", {})
        daily = data.get("daily", {})
        
        # Get today's max rain probability
        rain_prob = daily.get("precipitation_probability_max", [0])[0] if daily else 0
        
        return {
            "temperature": curr.get("temperature_2m", 25.0),
            "humidity": curr.get("relative_humidity_2m", 60.0),
            "rain_probability": float(rain_prob),
            "is_raining": curr.get("rain", 0) > 0 or curr.get("precipitation", 0) > 0,
            "description": self._get_description(curr.get("rain", 0), rain_prob),
            "source": "Open-Meteo"
        }

    def _generate_fallback(self, previous: Dict = None) -> Dict[str, Any]:
        """Generate simulated tropical weather."""
        if not previous:
            return {
                "temperature": 28.0,
                "humidity": 70.0,
                "rain_probability": 20.0,
                "is_raining": False,
                "description": "Sunny (Simulated)",
                "source": "Simulation"
            }
        
        # Simple random walk
        temp = previous["temperature"] + random.uniform(-0.5, 0.5)
        humidity = max(30, min(100, previous["humidity"] + random.uniform(-2, 2)))
        
        # Rain logic
        is_raining = previous["is_raining"]
        if is_raining:
            if random.random() < 0.2: is_raining = False  # Stop raining
        else:
            if random.random() < 0.05: is_raining = True  # Start raining
            
        rain_prob = 90 if is_raining else 20
        
        return {
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "rain_probability": rain_prob,
            "is_raining": is_raining,
            "description": "Rainy (Simulated)" if is_raining else "Cloudy (Simulated)",
            "source": "Simulation"
        }

    def _get_description(self, rain_mm: float, rain_prob: float) -> str:
        if rain_mm > 0: return "Raining"
        if rain_prob > 60: return "Overcast"
        if rain_prob > 30: return "Partly Cloudy"
        return "Sunny"
