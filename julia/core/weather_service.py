"""
julia.core.weather_service — OpenWeather API integration.

Fetches weather forecasts for watering decisions.
Includes caching and offline fallback.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from julia.core.config import JuliaConfig
from julia.core.decision_engine import WeatherForecast

logger = logging.getLogger("julia.weather")


class WeatherService:
    """
    Fetches weather data from the OpenWeather API.

    Features:
    - Caches results to avoid excessive API calls
    - Falls back gracefully when offline
    - Extracts rain probability from forecast data
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, config: JuliaConfig, db=None):
        self.api_key = config.weather.api_key
        self.lat = config.weather.latitude
        self.lon = config.weather.longitude
        self.cache_ttl = config.weather.cache_ttl_minutes * 60  # Convert to seconds
        self._db = db  # Optional JuliaDatabase for persistent weather logging
        self._cache: Optional[dict] = None
        self._cache_time: float = 0
        self._cache_file = Path(config.config_path).parent / "weather_cache.json"

    def get_forecast(self) -> WeatherForecast:
        """
        Get the current weather forecast.

        Returns a WeatherForecast dataclass. If the API is unavailable,
        returns a default PR-appropriate forecast marked as unavailable.
        """
        if not self.api_key:
            logger.info("No OpenWeather API key — using defaults.")
            return WeatherForecast.unavailable()

        # Check cache
        cached = self._get_cached()
        if cached:
            return cached

        # Fetch fresh data
        try:
            forecast = self._fetch_forecast()
            self._save_cache(forecast)
            return forecast
        except Exception as e:
            logger.warning(f"Weather API failed: {e}. Using fallback.")
            # Try loading stale cache
            stale = self._load_stale_cache()
            if stale:
                return stale
            return WeatherForecast.unavailable()

    def _fetch_forecast(self) -> WeatherForecast:
        """Fetch 5-day/3-hour forecast from OpenWeather."""
        resp = requests.get(
            f"{self.BASE_URL}/forecast",
            params={
                "lat": self.lat,
                "lon": self.lon,
                "appid": self.api_key,
                "units": "metric",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse the forecast entries
        entries = data.get("list", [])
        if not entries:
            return WeatherForecast.unavailable()

        # Calculate rain probability for next 24h and 48h
        rain_probs_24h = []
        rain_probs_48h = []
        temps = []

        for entry in entries:
            dt = entry.get("dt", 0)
            hours_from_now = (dt - time.time()) / 3600

            # Pop probability (0-1 from API, we store as 0-100)
            pop = entry.get("pop", 0) * 100

            if 0 <= hours_from_now <= 24:
                rain_probs_24h.append(pop)
            if 0 <= hours_from_now <= 48:
                rain_probs_48h.append(pop)

            # Temperature
            main = entry.get("main", {})
            temps.append(main.get("temp", 25))

        # Use max rain probability (worst case)
        rain_24h = max(rain_probs_24h) if rain_probs_24h else 0
        rain_48h = max(rain_probs_48h) if rain_probs_48h else 0
        temp_high = max(temps) if temps else 30
        temp_low = min(temps) if temps else 22
        avg_humidity = data.get("list", [{}])[0].get("main", {}).get("humidity", 70)

        # Description from current weather
        description = "Clear"
        if entries:
            weather_list = entries[0].get("weather", [])
            if weather_list:
                description = weather_list[0].get("description", "Clear").capitalize()

        forecast = WeatherForecast(
            rain_probability_24h=rain_24h,
            rain_probability_48h=rain_48h,
            temp_high=temp_high,
            temp_low=temp_low,
            humidity=avg_humidity,
            description=description,
            is_available=True,
        )

        logger.info(
            f"🌤️ Weather: {description}, Rain 24h: {rain_24h:.0f}%, "
            f"Temp: {temp_low:.0f}-{temp_high:.0f}°C"
        )

        # Persist to database if available
        if self._db:
            try:
                self._db.log_weather_snapshot(
                    temperature=avg_humidity,  # Current temp not directly available
                    humidity=avg_humidity,
                    rain_24h=rain_24h,
                    rain_48h=rain_48h,
                    temp_high=temp_high,
                    temp_low=temp_low,
                    description=description,
                )
            except Exception as e:
                logger.debug(f"Failed to log weather to DB: {e}")

        return forecast

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _get_cached(self) -> Optional[WeatherForecast]:
        """Return cached forecast if still fresh."""
        if self._cache and (time.time() - self._cache_time) < self.cache_ttl:
            logger.debug("Using cached weather data.")
            return self._dict_to_forecast(self._cache)
        return None

    def _save_cache(self, forecast: WeatherForecast):
        """Save forecast to memory and disk cache."""
        self._cache = {
            "rain_probability_24h": forecast.rain_probability_24h,
            "rain_probability_48h": forecast.rain_probability_48h,
            "temp_high": forecast.temp_high,
            "temp_low": forecast.temp_low,
            "humidity": forecast.humidity,
            "description": forecast.description,
        }
        self._cache_time = time.time()

        # Persist to disk for stale fallback
        try:
            cache_data = {**self._cache, "cached_at": self._cache_time}
            self._cache_file.write_text(json.dumps(cache_data, indent=2))
        except OSError as e:
            logger.debug(f"Failed to write weather cache: {e}")

    def _load_stale_cache(self) -> Optional[WeatherForecast]:
        """Load cached forecast from disk, even if stale."""
        try:
            if self._cache_file.exists():
                data = json.loads(self._cache_file.read_text())
                logger.info("Using stale cached weather data as fallback.")
                forecast = self._dict_to_forecast(data)
                forecast.description += " (cached — may be stale)"
                return forecast
        except (json.JSONDecodeError, OSError):
            pass
        return None

    @staticmethod
    def _dict_to_forecast(data: dict) -> WeatherForecast:
        """Convert a cache dict back to a WeatherForecast."""
        return WeatherForecast(
            rain_probability_24h=data.get("rain_probability_24h", 0),
            rain_probability_48h=data.get("rain_probability_48h", 0),
            temp_high=data.get("temp_high", 30),
            temp_low=data.get("temp_low", 22),
            humidity=data.get("humidity", 70),
            description=data.get("description", "unknown"),
            is_available=True,
        )
