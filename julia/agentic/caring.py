from enum import Enum
from .state import AgenticContext, WorldState

class AgentState(Enum):
    NORMAL = "NORMAL"         # Care level 1
    SUPPORTIVE = "SUPPORTIVE" # Care level 2
    STORM_PREP = "STORM_PREP" # Care level 3
    RECOVERY = "RECOVERY"     # Care level 2-3 (declining)

class CaringLayer:
    """
    Determines Julia's emotional state and care level based on 
    the external WorldState and internal memory of mistakes.
    """
    
    def determine_state(self, world_state: WorldState, past_mistakes=None) -> AgentState:
        """
        Determines the current state. Top priority is STORM_PREP,
        followed by RECOVERY, then SUPPORTIVE, then NORMAL.
        """
        if self.is_storm_threat(world_state):
            return AgentState.STORM_PREP
        
        if self.is_recovering(world_state):
            return AgentState.RECOVERY
        
        if self.needs_support(world_state, past_mistakes):
            return AgentState.SUPPORTIVE
        
        return AgentState.NORMAL
    
    def is_storm_threat(self, world_state: WorldState) -> bool:
        return (
            world_state.risk_weekly > 0.6 or
            world_state.storm_proximity_km < 500 or
            (world_state.hurricane_season and world_state.disturbance_active)
        )
    
    def is_recovering(self, world_state: WorldState) -> bool:
        return (
            world_state.hours_since_last_storm < 120 or   # 5 days
            world_state.hours_since_drought_end < 72 or    # 3 days
            world_state.recent_plant_stress_event
        )
    
    def needs_support(self, world_state: WorldState, past_mistakes=None) -> bool:
        # Check if user needs support due to past failures or elevated risk
        has_similar_failure = False
        if past_mistakes and hasattr(past_mistakes, 'has_similar_failure'):
            has_similar_failure = past_mistakes.has_similar_failure(world_state.conditions_hash())
        elif isinstance(past_mistakes, list) and past_mistakes:
            # Simple list fallback
            has_similar_failure = any(
                m.get('conditions_hash') == world_state.conditions_hash() 
                for m in past_mistakes
            )
            
        return (
            has_similar_failure or
            world_state.risk_weekly > 0.4
        )

    def calculate_care_level(self, context: AgenticContext) -> int:
        """
        Care level = max of uncertainty, storm risk, and failure history.
        Cap at 3.
        """
        uncertainty_score = 2 if context.user_uncertainty_detected else 0
        
        # Risk > 0.6 equals 3, Risk > 0.4 equals 2, Risk > 0.2 equals 1
        if context.world_state.risk_weekly > 0.6:
            storm_risk_score = 3
        elif context.world_state.risk_weekly > 0.4:
            storm_risk_score = 2
        elif context.world_state.risk_weekly > 0.2:
            storm_risk_score = 1
        else:
            storm_risk_score = 0
            
        # Failure score
        failure_score = 0
        if context.past_mistakes:
            if hasattr(context.past_mistakes, 'has_similar_failure'):
                if context.past_mistakes.has_similar_failure(context.conditions_hash()):
                    failure_score = 2
            elif isinstance(context.past_mistakes, list) and context.past_mistakes:
                if any(m.get('conditions_hash') == context.conditions_hash() for m in context.past_mistakes):
                    failure_score = 2
                    
        care_level = max(uncertainty_score, storm_risk_score, failure_score)
        
        # Default minimum care is 1 unless perfectly optimal
        if care_level == 0 and context.world_state.risk_weekly > 0:
            care_level = 1
            
        return min(care_level, 3)
