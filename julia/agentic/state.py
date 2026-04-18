import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class WorldState:
    """A unified snapshot of all sensor, weather, and context data at a given moment."""
    
    # Real-time signals
    soil_moisture: Dict[str, float] = field(default_factory=dict)
    temperature: float = 25.0
    humidity: float = 60.0
    light_level: float = 500.0
    
    # Short-term context (hours/days)
    weather_forecast: Dict[str, Any] = field(default_factory=dict)
    rain_probability_24h: float = 0.0
    storm_proximity_km: float = 9999.0
    disturbance_active: bool = False
    
    # Medium-term context (weeks/months)
    risk_weekly: float = 0.0
    risk_monthly: float = 0.0
    enso_phase: str = "Neutral"
    hurricane_season: bool = False
    
    # Long-term context (seasons/years)
    historical_rainfall: float = 0.0
    crop_resilience: float = 1.0
    drought_active: bool = False
    drought_cycle_position: float = 0.0
    
    # Temporal context
    timestamp: datetime = field(default_factory=datetime.now)
    season: str = "Dry"
    days_since_last_water: Dict[str, float] = field(default_factory=dict)
    
    # Day length (hours)
    day_length_hours: float = 12.0
    
    # Events tracking (for RECOVERY state)
    hours_since_last_storm: float = 9999.0
    hours_since_drought_end: float = 9999.0
    recent_plant_stress_event: bool = False
    
    @property
    def signal_count(self) -> int:
        """Rough estimate of how many data streams fed into this state."""
        return len(self.soil_moisture) + 6  # temp, hum, light, rain, enso, risk
    
    def conditions_hash(self) -> str:
        """
        Creates a deterministic hash of the critical conditions to match against past mistakes.
        We bucket continuous values so similar states hash to the same value.
        """
        # Bucket temperature to nearest 2 degrees
        temp_bucket = round(self.temperature / 2) * 2
        # Bucket rain probability to nearest 20%
        rain_bucket = round(self.rain_probability_24h / 20) * 20
        # Bucket risk to nearest 0.1
        risk_bucket = round(self.risk_weekly, 1)
        
        signature = f"T{temp_bucket}_R{rain_bucket}_Risk{risk_bucket}_{self.enso_phase}"
        if self.hurricane_season and self.disturbance_active:
            signature += "_STORM"
        if self.drought_active:
            signature += "_DROUGHT"
            
        return hashlib.md5(signature.encode()).hexdigest()[:8]

@dataclass
class AgenticContext:
    """The working memory structure passed through the OODA-L loop."""
    world_state: WorldState
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    detected_patterns: List[str] = field(default_factory=list)
    past_mistakes: Any = None  # Will be a MistakeList or similar that supports has_similar_failure()
    confidence: str = "MODERATE"
    agent_state: Any = None  # Will be AgentState Enum
    care_level: int = 1
    reasoning_chain: List[str] = field(default_factory=list)
    user_uncertainty_detected: bool = False
    
    def conditions_hash(self) -> str:
        """Pass-through to world state hash for convenience."""
        return self.world_state.conditions_hash()
    
    def has_recent_plant_stress_event(self) -> bool:
        """Pass-through to world state flags."""
        return self.world_state.recent_plant_stress_event
    
    def snapshot(self) -> str:
        """Returns a JSON string of the context for the Bitacora/Mistake Memory."""
        return json.dumps({
            "timestamp": self.world_state.timestamp.isoformat(),
            "temperature": self.world_state.temperature,
            "rain_prob": self.world_state.rain_probability_24h,
            "risk_weekly": self.world_state.risk_weekly,
            "agent_state": getattr(self.agent_state, "value", str(self.agent_state)),
            "care_level": self.care_level,
            "hash": self.conditions_hash()
        })
