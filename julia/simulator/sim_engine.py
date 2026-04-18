"""
julia.simulator.sim_engine — Virtual environment simulation.

Models realistic soil moisture, weather, and plant health over time.
Used to test Julia's decision engine without real hardware.

Physics:
- Soil moisture drops via evaporation (faster in heat, lower humidity)
- Watering adds moisture (capped at 100%)
- Temperature follows a sinusoidal daily cycle
- Rain events happen stochastically and add moisture
- Plant health degrades from chronic over/under watering
"""

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from enum import Enum

from julia.core.config import JuliaConfig
from julia.data.database import JuliaDatabase
from julia.core.weather import WeatherService
from julia.core.brain import Brain, Decision
from julia.core.llm_config import get_llm_config
from julia.core.llm_brain import JuliaBrain
from julia.vision.camera import CameraService
from julia.sensors.sensor_reader import SensorData


# ======================================================================
# Simulation Models
# ======================================================================

@dataclass
class SimWeather:
    """Simulated weather state."""
    temperature: float = 28.0
    humidity: float = 72.0
    is_raining: bool = False
    rain_intensity: float = 0.0       # 0-1
    rain_probability_24h: float = 30.0
    cloud_cover: float = 0.3          # 0-1
    wind_speed: float = 8.0           # km/h
    description: str = "Partly Cloudy"


@dataclass
class SimPlant:
    """Simulated plant state."""
    plant_id: str
    name: str
    emoji: str
    soil_moisture: float = 55.0
    health: float = 100.0             # 0-100
    health_status: str = "healthy"
    growth_stage: str = "growing"
    days_alive: int = 0
    times_watered: int = 0
    total_water_ml: int = 0
    stress_hours: int = 0             # Hours outside optimal range


@dataclass
class SimEvent:
    """A logged event in the simulation."""
    sim_time: str
    real_time: str
    event_type: str          # "decision", "watering", "rain", "health", "alert"
    plant_id: str
    message: str
    icon: str = ""


class SimulationState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class SimulationEngine:
    """
    Virtual environment for Julia.

    Simulates realistic Puerto Rico garden conditions
    and runs Julia's decision engine against them.
    """

    # Puerto Rico climate parameters
    BASE_TEMP = 27.0          # Average temp °C
    TEMP_AMPLITUDE = 5.0      # Daily swing ±°C
    BASE_HUMIDITY = 72.0
    EVAPORATION_RATE = 0.8    # % moisture loss per sim-hour (base)
    RAIN_BASE_CHANCE = 0.08   # Per-hour chance of rain starting
    RAIN_DURATION_HOURS = 2   # Average rain duration

    def __init__(self, config: Optional[JuliaConfig] = None):
        self.config = config or JuliaConfig()
        
        # Core Components
        self.db = JuliaDatabase()
        self.weather_service = WeatherService()
        self.brain = Brain() # Rule-based
        self.llm_config = get_llm_config()
        self.llm_brain = JuliaBrain(
            base_url=self.llm_config.base_url,
            model=self.llm_config.model,
            db=self.db
        )
        self.camera_service = CameraService()

        # Simulation state
        self.state = SimulationState.STOPPED
        self.speed = 1.0              # Sim-hours per real-second
        self.sim_time = datetime(2026, 2, 18, 6, 0, tzinfo=timezone.utc)  # Start at 6 AM
        self.elapsed_hours = 0.0

        # World state
        self.weather = SimWeather()
        self.plants: Dict[str, SimPlant] = {}
        self.events: List[SimEvent] = []
        self.decision_history: List[dict] = []

        # Chart data (time series)
        self.chart_data: Dict[str, List[dict]] = {}

        # Timing
        self._last_decision_hour = -6  # Force first check immediately
        self._rain_end_hour = 0
        self._last_update = time.time()

        # Initialize plants
        self._init_plants()

    def _init_plants(self):
        """Create virtual plants from config."""
        starter_plants = {
            "basil": (55.0, "🌿"),
            "pepper": (48.0, "🌶️"),
            "tomato": (62.0, "🍅"),
        }
        for plant_id, (moisture, emoji) in starter_plants.items():
            profile = self.config.get_profile(plant_id)
            self.plants[plant_id] = SimPlant(
                plant_id=plant_id,
                name=profile.name,
                emoji=emoji,
                soil_moisture=moisture + random.uniform(-5, 5),
            )
            self.chart_data[plant_id] = []
        
        # Load last 24h of data
        self._load_history()

    def _load_history(self):
        """Load historical sensor data from DB."""
        for plant_id in self.plants:
            # Get last 24h
            history = self.db.get_sensor_trend(plant_id, hours=24)
            for point in history:
                # Convert DB timestamp "YYYY-MM-DD HH:MM:SS" to chart label
                try:
                    dt = datetime.strptime(point['timestamp'], "%Y-%m-%d %H:%M:%S")
                    label = dt.strftime("%b %d %H:%M")
                except ValueError:
                     label = point['timestamp']

                self.chart_data[plant_id].append({
                    "time": label,
                    "moisture": point['soil_moisture'],
                    "temperature": point.get('temperature', 0),
                    "humidity": point.get('humidity', 0),
                    "health": 100, 
                    "is_raining": False, 
                })

    # ------------------------------------------------------------------
    # Main Tick
    # ------------------------------------------------------------------

    def tick(self) -> float:
        """
        Advance the simulation by one time step.

        Returns the number of sim-hours elapsed this tick.
        """
        if self.state != SimulationState.RUNNING:
            return 0.0

        now = time.time()
        real_dt = now - self._last_update
        self._last_update = now

        # Calculate sim time elapsed
        sim_hours = real_dt * self.speed
        if sim_hours > 1.0:
            sim_hours = 1.0  # Cap per-tick to prevent jumps

        self.elapsed_hours += sim_hours
        self.sim_time += timedelta(hours=sim_hours)

        # Update world
        self._update_weather(sim_hours)
        self._update_soil(sim_hours)
        self._update_health(sim_hours)
        self._record_chart_data()

        # Run Julia's decision engine every 6 sim-hours
        if self.elapsed_hours - self._last_decision_hour >= 6:
            self._run_decision_cycle()
            self._last_decision_hour = self.elapsed_hours

        return sim_hours

    # ------------------------------------------------------------------
    # Weather Simulation (Real + Fallback)
    # ------------------------------------------------------------------

    def _update_weather(self, dt_hours: float):
        """Update weather using WeatherService (Real API or Simulation)."""
        # Only query API every 1 sim-hour to avoid spamming
        current_sim_hour = int(self.elapsed_hours)
        last_update_hour = getattr(self, "_last_weather_update_hour", -1)
        
        if current_sim_hour > last_update_hour:
            data = self.weather_service.update()
            self._last_weather_update_hour = current_sim_hour
            
            # Update Simulation Weather State
            self.weather.temperature = data["temperature"]
            self.weather.humidity = data["humidity"]
            self.weather.rain_probability_24h = data["rain_probability"]
            self.weather.is_raining = data["is_raining"]
            self.weather.description = data.get("description", "Unknown")
            
            # Log Snapshot to DB
            self.db.log_weather_snapshot(
                temperature=self.weather.temperature,
                humidity=self.weather.humidity,
                rain_24h=self.weather.rain_probability_24h,
                rain_48h=self.weather.rain_probability_24h,
                temp_high=self.weather.temperature + 3.0,
                temp_low=self.weather.temperature - 3.0,
                description=self.weather.description,
                is_available=True,
            )
            
            # Handle Rain Event Logging
            if self.weather.is_raining and not getattr(self, "_was_raining", False):
                 self._add_event("rain", "", f"🌧️ Rain started ({data['source']})", "🌧️")
            elif not self.weather.is_raining and getattr(self, "_was_raining", False):
                 self._add_event("weather", "", "🌤️ Rain stopped", "☀️")
            
            self._was_raining = self.weather.is_raining

        # Micro-updates for variation between API calls (Physics)
        pass 

    # ------------------------------------------------------------------
    # Soil Simulation
    # ------------------------------------------------------------------

    def _update_soil(self, dt_hours: float):
        """Update soil moisture for all plants."""
        for plant in self.plants.values():
            # Evaporation (faster when hot, dry air)
            temp_factor = max(0.5, (self.weather.temperature - 20) / 15)
            humid_factor = max(0.3, 1 - (self.weather.humidity / 100))
            evaporation = self.EVAPORATION_RATE * temp_factor * humid_factor * dt_hours
            plant.soil_moisture -= evaporation

            # Rain adds moisture
            if self.weather.is_raining:
                rain_amount = self.weather.rain_intensity * 3.0 * dt_hours
                plant.soil_moisture += rain_amount

            # Clamp
            plant.soil_moisture = max(0, min(100, round(plant.soil_moisture, 1)))

    # ------------------------------------------------------------------
    # Health Simulation
    # ------------------------------------------------------------------

    def _update_health(self, dt_hours: float):
        """Update plant health based on conditions."""
        for plant in self.plants.values():
            profile = self.config.get_profile(plant.plant_id)

            # Check if in optimal range
            if profile.min_moisture <= plant.soil_moisture <= profile.max_moisture:
                # Healthy — slowly recover
                plant.health = min(100, plant.health + 0.3 * dt_hours)
                plant.health_status = "healthy"
            elif plant.soil_moisture < profile.min_moisture:
                # Too dry — stress
                severity = (profile.min_moisture - plant.soil_moisture) / profile.min_moisture
                plant.health -= severity * 2.0 * dt_hours
                plant.stress_hours += 1
                if plant.soil_moisture < profile.min_moisture * 0.5:
                    plant.health_status = "wilting"
                else:
                    plant.health_status = "underwatered"
            elif plant.soil_moisture > profile.max_moisture:
                # Too wet — slower stress
                severity = (plant.soil_moisture - profile.max_moisture) / (100 - profile.max_moisture + 0.01)
                plant.health -= severity * 1.0 * dt_hours
                plant.health_status = "overwatered"

            plant.health = max(0, min(100, round(plant.health, 1)))

            # Alert on critical health
            if plant.health < 30 and plant.health_status != "healthy":
                self._add_event(
                    "alert", plant.plant_id,
                    f"⚠️ {plant.emoji} {plant.name} health critical: {plant.health:.0f}%!",
                    "🚨"
                )

    # ------------------------------------------------------------------
    # Decision Cycle
    # ------------------------------------------------------------------

    def _run_decision_cycle(self):
        """Run Julia's BRAIN on simulated data."""
        
        for plant in self.plants.values():
            profile = self.config.get_profile(plant.plant_id)

            # 1. Log Sensor Reading
            self.db.log_sensor_reading(
                plant.plant_id, 
                plant.soil_moisture, 
                self.weather.temperature, 
                self.weather.humidity
            )

            # 2. Ask the Brain
            decision = self.brain.decide(
                plant.plant_id,
                plant.soil_moisture,
                profile.min_moisture,
                self.weather.rain_probability_24h
            )

            # 3. Log Decision
            self.db.log_decision(
                plant_id=plant.plant_id,
                decision=decision.action,
                water_amount_ml=200 if decision.action == "WATER" else 0,
                reason=decision.reason,
                confidence=1.0,
                soil_moisture=plant.soil_moisture,
                temperature=self.weather.temperature,
                humidity=self.weather.humidity,
                rain_probability=self.weather.rain_probability_24h,
                weather_available=True,
            )

            # 4. Update History (In-Memory for Dashboard)
            self.decision_history.append({
                "sim_time": self.sim_time.isoformat(),
                "plant_id": plant.plant_id,
                "decision": decision.action,
                "reason": decision.reason,
                "moisture": plant.soil_moisture,
            })

            # 5. Execute Action
            icon = "⏸️"
            if decision.action == "WATER":
                amount_ml = 200 # Standard dose
                
                # Execute physics
                moisture_increase = amount_ml * 0.08
                plant.soil_moisture = min(100, plant.soil_moisture + moisture_increase)
                plant.times_watered += 1
                plant.total_water_ml += amount_ml
                
                icon = "💧"
                self._add_event(
                    "watering", plant.plant_id,
                    f"💧 Watered {plant.emoji} {plant.name} ({decision.reason})",
                    "💧"
                )
            else:
                 self._add_event("decision", plant.plant_id, decision.reason, icon)

    # ------------------------------------------------------------------
    # Chart Data
    # ------------------------------------------------------------------

    def _record_chart_data(self):
        """Record a data point for charts."""
        time_label = self.sim_time.strftime("%b %d %H:%M")
        for plant in self.plants.values():
            self.chart_data[plant.plant_id].append({
                "time": time_label,
                "moisture": round(plant.soil_moisture, 1),
                "temperature": round(self.weather.temperature, 1),
                "humidity": round(self.weather.humidity, 1),
                "health": round(plant.health, 1),
                "is_raining": self.weather.is_raining,
            })
            # Keep last 200 points per plant
            if len(self.chart_data[plant.plant_id]) > 200:
                self.chart_data[plant.plant_id] = self.chart_data[plant.plant_id][-200:]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _add_event(self, event_type: str, plant_id: str, message: str, icon: str = ""):
        """Add an event to the log."""
        self.events.append(SimEvent(
            sim_time=self.sim_time.strftime("%b %d %H:%M"),
            real_time=datetime.now().strftime("%H:%M:%S"),
            event_type=event_type,
            plant_id=plant_id,
            message=message,
            icon=icon,
        ))
        # Keep last 100 events
        if len(self.events) > 100:
            self.events = self.events[-100:]

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def start(self):
        """Start or resume simulation."""
        self.state = SimulationState.RUNNING
        self._last_update = time.time()

    def pause(self):
        """Pause simulation."""
        self.state = SimulationState.PAUSED

    def stop(self):
        """Stop and reset simulation."""
        self.state = SimulationState.STOPPED

    def set_speed(self, speed: float):
        """Set simulation speed (sim-hours per real-second)."""
        self.speed = max(0.1, min(24.0, speed))

    def manual_water(self, plant_id: str, amount_ml: int = 200):
        """Manually water a plant in the simulation."""
        plant = self.plants.get(plant_id)
        if plant:
            moisture_increase = amount_ml * 0.08
            plant.soil_moisture = min(100, plant.soil_moisture + moisture_increase)
            plant.times_watered += 1
            plant.total_water_ml += amount_ml
            self._add_event(
                "watering", plant_id,
                f"🖐️ Manual watering: {plant.emoji} {plant.name} +{amount_ml}ml → {plant.soil_moisture:.0f}%",
                "🖐️"
            )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def get_latest_image(self) -> bytes:
        """Get the latest camera frame."""
        return self.camera_service.capture()

    # ------------------------------------------------------------------
    # API Data
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Get full simulation state for the dashboard."""
        return {
            "status": self.state.value,
            "speed": self.speed,
            "sim_time": self.sim_time.strftime("%A, %b %d %Y — %I:%M %p"),
            "sim_day": int(self.elapsed_hours / 24),
            "sim_hour": round(self.elapsed_hours % 24, 1),
            "elapsed_hours": round(self.elapsed_hours, 1),
            "weather": {
                "temperature": self.weather.temperature,
                "humidity": self.weather.humidity,
                "is_raining": self.weather.is_raining,
                "rain_probability": round(self.weather.rain_probability_24h, 0),
                "description": self.weather.description,
                "wind_speed": round(self.weather.wind_speed, 1),
            },
            "plants": {
                pid: {
                    "name": p.name,
                    "emoji": p.emoji,
                    "soil_moisture": p.soil_moisture,
                    "health": p.health,
                    "health_status": p.health_status,
                    "times_watered": p.times_watered,
                    "total_water_ml": p.total_water_ml,
                    "min_moisture": self.config.get_profile(pid).min_moisture,
                    "max_moisture": self.config.get_profile(pid).max_moisture,
                    "optimal_moisture": self.config.get_profile(pid).optimal_moisture,
                }
                for pid, p in self.plants.items()
            },
            "chart_data": self.chart_data,
            "events": [
                {
                    "sim_time": e.sim_time,
                    "type": e.event_type,
                    "plant_id": e.plant_id,
                    "message": e.message,
                    "icon": e.icon,
                }
                for e in reversed(self.events[-30:])  # Last 30, newest first
            ],
            "stats": {
                "total_decisions": len(self.decision_history),
                "total_waterings": sum(p.times_watered for p in self.plants.values()),
                "total_water_ml": sum(p.total_water_ml for p in self.plants.values()),
                "avg_health": round(
                    sum(p.health for p in self.plants.values()) / max(1, len(self.plants)), 1
                ),
            },
        }
