import logging
from typing import Dict, Any, List
from datetime import datetime

from .state import WorldState, AgenticContext
from .caring import CaringLayer, AgentState
from .memory import MistakeMemory

logger = logging.getLogger("julia.agentic.context_engine")

class ContextEngine:
    """
    ORIENT phase of the OODA-L loop.
    Interprets WorldState using memory, patterns, and the CaringLayer's empathy.
    """
    
    def __init__(self, db, memory=None, caring_layer=None):
        self.db = db
        self.memory = memory or MistakeMemory(db)
        self.caring = caring_layer or CaringLayer()
        
    def build_context(self, state: WorldState) -> AgenticContext:
        """
        Takes raw WorldState and builds the intelligent AgenticContext.
        """
        # 1. Fetch recent events (last 48 hours for short-term memory)
        recent_events = self._fetch_recent_events()
        
        # 2. Check for past mistakes under these exact conditions
        past_mistakes = self.memory.get_mistakes(state.conditions_hash())
        
        # 3. Simple Pattern Detection 
        patterns = self._detect_patterns(state, past_mistakes)
        
        # 4. CARING: Determine Agent State and Care Budget
        agent_state = self.caring.determine_state(state, past_mistakes)
        
        # Temporary context to calculate care level
        temp_ctx = AgenticContext(world_state=state, past_mistakes=past_mistakes)
        temp_ctx.agent_state = agent_state
        temp_ctx.user_uncertainty_detected = False # Could be parsed from LLM in future
        
        care_level = self.caring.calculate_care_level(temp_ctx)
        
        # 5. Calculate Confidence Score
        confidence = self.calculate_confidence(state, past_mistakes)
        
        return AgenticContext(
            world_state=state,
            recent_events=recent_events,
            detected_patterns=patterns,
            past_mistakes=past_mistakes,
            confidence=confidence,
            agent_state=agent_state,
            care_level=care_level,
            reasoning_chain=[],
            user_uncertainty_detected=False
        )
        
    def _fetch_recent_events(self) -> List[Dict[str, Any]]:
        """Fetch last 48 hours of decisions to understand recent actions."""
        if not self.db:
            return []
        try:
            return self.db.get_decision_history(days=2)
        except Exception:
            return []
            
    def _detect_patterns(self, state: WorldState, past_mistakes: List[Dict]) -> List[str]:
        """Simple pattern detection based on state arrays."""
        patterns = []
        if state.drought_active and state.temperature > 32.0:
            patterns.append("Extreme Heat + Drought Cycle")
        if state.risk_weekly > 0.6 and state.rain_probability_24h > 80:
            patterns.append("Imminent Heavy Rainfall Event")
        if len(past_mistakes) >= 3:
            patterns.append("Historically Error-Prone Condition")
        return patterns
        
    def calculate_confidence(self, state: WorldState, past_mistakes: List[Dict]) -> str:
        """
        Calculates how confident Julia is in her understanding of the situation.
        Low confidence triggers invariant protections (asking clarification).
        """
        aligned_signals = 0
        conflicting = False
        
        # Are sensors and weather aligned?
        if state.risk_weekly > 0.5 and state.rain_probability_24h > 50:
            aligned_signals += 1
        elif state.risk_weekly > 0.5 and state.rain_probability_24h < 10:
            conflicting = True
            
        if state.drought_active and state.soil_moisture:
            avg_moisture = sum(state.soil_moisture.values()) / len(state.soil_moisture)
            if avg_moisture < 40:
                aligned_signals += 1
            elif avg_moisture > 70:
                conflicting = True # Drought but soil is very wet
                
        # Has she failed here before?
        if len(past_mistakes) > 0:
            aligned_signals -= 1 # Reduces confidence
            
        # Compile score
        if conflicting or aligned_signals < 0:
            return "LOW (0.4)"
        elif aligned_signals > 1:
            return "HIGH (0.9)"
        else:
            return "MODERATE (0.7)"
