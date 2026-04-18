from datetime import datetime
import json
import logging
from typing import Dict, Any, List

from .state import WorldState
from julia.core.weather_service import WeatherService
from julia.core.risk_engine import ClimateRiskEngine

logger = logging.getLogger("julia.agentic.perception")

class PerceptionLayer:
    """
    OBSERVE phase of the OODA-L loop.
    Fuses all sensor, API, and historical data into a unified WorldState snapshot.
    """
    
    def __init__(self, db, weather_service=None, risk_engine=None):
        self.db = db
        # We allow injecting mocks for testing
        self.weather = weather_service or WeatherService(api_key="dry_run", use_nws=True)
        self.risk_engine = risk_engine or ClimateRiskEngine(db=db)
        
    def get_world_state(self) -> WorldState:
        """Gathers all signals and builds the WorldState."""
        state = WorldState()
        
        # 1. Sensors (Simulated via DB reading for now)
        try:
            if self.db:
                # Get latest reading for each active plant
                plants = ["basil", "pepper", "tomato"]
                for p in plants:
                    readings = self.db.get_sensor_trend(p, hours=1)
                    if readings:
                        latest = readings[-1]
                        state.soil_moisture[p] = latest.get("soil_moisture", 50.0)
                        # We just take the last plant's ambient data as general ambient
                        state.temperature = latest.get("temperature", 25.0)
                        state.humidity = latest.get("humidity", 60.0)
        except Exception as e:
            logger.warning(f"Perception failed to read sensors: {e}")
            
        # 2. Weather & Storms
        try:
            forecast = self.weather.get_forecast_48h()
            if forecast and len(forecast) > 0:
                next_24h = forecast[:8] # Assuming 3hr blocks
                state.rain_probability_24h = max([f.get("pop", 0) * 100 for f in next_24h])
                state.weather_forecast = {"next_24h": next_24h}
                
                # Check for active disturbances / hurricanes in forecast
                for f in forecast:
                    desc = f.get("weather", [{}])[0].get("description", "").lower()
                    if any(word in desc for word in ["hurricane", "tropical storm", "cyclone", "depression"]):
                        state.disturbance_active = True
                        state.storm_proximity_km = 300.0 # Heuristic if in forecast
                        break
        except Exception as e:
            logger.warning(f"Perception failed to read weather: {e}")
            
        # 3. Risk Engine (Medium-term context)
        try:
            features = self._build_features(state)
            v1_scores = self.risk_engine.evaluate_v1(features)
            state.risk_weekly = v1_scores.get("risk_weekly_rf", 0.0)
            state.risk_monthly = v1_scores.get("risk_monthly_gbc", 0.0)
            state.enso_phase = features.get("enso_phase", "Neutral")
        except Exception as e:
            logger.warning(f"Perception failed to compute risk: {e}")
            
        # 4. Temporal & Calendar Context
        now = datetime.now()
        month = now.month
        state.hurricane_season = 6 <= month <= 11
        state.season = "Wet" if (4 <= month <= 5 or 8 <= month <= 11) else "Dry"
        
        # 5. Events Tracking (from DB)
        if self.db:
            try:
                # Rough heuristics for recovery tracking
                # In a real system, we'd query explicit event tables
                recent_risk = self.db.get_recent_weather(hours=120)  # past 5 days
                if recent_risk:
                    for w in recent_risk:
                        if w.get("rain_probability_24h", 0) > 80: # major storm marker
                            diff = now - datetime.fromisoformat(w["timestamp"])
                            state.hours_since_last_storm = min(state.hours_since_last_storm, diff.total_seconds() / 3600)
            except Exception:
                pass
                
        return state
        
    def _build_features(self, state: WorldState) -> Dict[str, Any]:
        """Convert current perceived state into features for the Risk Engine."""
        return {
            "drought_index": -1.0 if state.drought_active else 0.5,
            "rain_anomaly_percent": 0.0,
            "enso_phase": "Neutral",
            "storm_proximity_km": state.storm_proximity_km,
            "storm_vmax_knots": 50 if state.disturbance_active else 0,
            "storm_count_30d": 0,
            "day_length_mins": int(state.day_length_hours * 60)
        }
