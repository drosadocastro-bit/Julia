import logging
from typing import List, Dict, Any

from julia.core.decision_engine import JuliaDecisionEngine
from julia.core.decision_engine import WeatherForecast
from julia.sensors.sensor_reader import SensorData
from .state import AgenticContext
from .caring import AgentState
from .invariants import ActionPlan, Action, Recommendation, CaringInvariants

logger = logging.getLogger("julia.agentic.planner")

class AgenticPlanner:
    """
    DECIDE phase of the OODA-L loop.
    Wraps the baseline DecisionEngine to apply Care Levels, Instincts, and Invariants.
    """
    
    def __init__(self, decision_engine=None):
        self.base_engine = decision_engine
        self.invariants = CaringInvariants()
        
    def plan(self, context: AgenticContext) -> ActionPlan:
        """
        Generates an ActionPlan based on the AgenticContext, fully validated.
        """
        # 1. Base deterministic evaluation (What would Phase 1 Julia do?)
        # For the architecture stub, we abstract the actual plant-by-plant loop 
        # and assume the base engine evaluates the primary context.
        try:
            # We must map WorldState into Phase 1 primitive types
            avg_moisture = sum(context.world_state.soil_moisture.values()) / len(context.world_state.soil_moisture) if context.world_state.soil_moisture else 50.0
            
            s_data = SensorData(
                sensor_id="basil",
                timestamp=str(context.world_state.timestamp) if hasattr(context.world_state, "timestamp") else "now",
                soil_moisture=avg_moisture,
                temperature=context.world_state.temperature,
                humidity=context.world_state.humidity,
                is_valid=True
            )
            
            w_data = WeatherForecast(
                rain_probability_24h=context.world_state.rain_probability_24h,
                rain_probability_48h=0.0,
                temp_high=context.world_state.temperature + 5,
                temp_low=context.world_state.temperature - 5,
                humidity=context.world_state.humidity,
                description="Mapped from Context",
                is_available=True
            )
            
            mock_risk = {
                "composite": {
                    "category": "CRITICAL" if context.world_state.risk_weekly >= 0.75 else "HIGH" if context.world_state.risk_weekly >= 0.50 else "LOW",
                    "final_advisory": "Extracted"
                }
            }
            
            # Phase 1 DecisionEngine expects: decide(plant_id, sensor_data, weather, plant_health, climate_risk)
            # We hardcode "basil" as the representative ID for the planner context evaluation
            base_result = self.base_engine.decide(
                plant_id="basil", 
                sensor_data=s_data, 
                weather=w_data, 
                climate_risk=mock_risk
            )
            
            base_decision = {
                "should_water": base_result.should_water(),
                "water_multiplier": 1.0, 
                "amount_ml": base_result.water_amount_ml,
                "reason": base_result.reason
            }
        except AttributeError:
            # Fallback if testing with a mock engine that only takes context
            base_decision = self.base_engine.decide(context)
        
        # Translate base decision dict into Action objects
        actions = []
        if base_decision.get("should_water"):
            actions.append(Action(
                type="WATER",
                reversible=False,
                multiplier=base_decision.get("water_multiplier", 1.0),
                amount_ml=base_decision.get("amount_ml", 0)
            ))
        else:
            actions.append(Action(type="DO_NOTHING", reversible=True))
            
        reasoning = [base_decision.get("reason", "Conditions nominal.")]

        
        # Initialize the ActionPlan
        plan = ActionPlan(
            actions=actions,
            reasoning=reasoning,
            confidence=context.confidence
        )
        
        # 2. Check for Emergency Overrides (Flash Drought / Extreme Storm)
        self._apply_emergency_overrides(plan, context)
        
        # 3. Apply Heuristic Instincts
        self._apply_instincts(plan, context)
        
        # 4. Modify plan based on Care Level
        self._apply_care_level(plan, context)
        
        # 5. INVARIANT GATE: Final check against the empathy firewall
        # For simplicity, we bundle the primary action into a Recommendation to test.
        rec = Recommendation(
            action=plan.actions[0] if plan.actions else Action(type="DO_NOTHING", reversible=True),
            reasoning=" ".join(plan.reasoning),
            monitor_signal="Check soil" if "monitor_signal" not in base_decision else base_decision.get("monitor_signal"),
            context_risk=context.world_state.risk_weekly,
            uncertainty=1.0 if "LOW" in context.confidence else 0.0,
            asks_clarification=plan.ask_confirmation
        )
        
        if not self.invariants.validate(rec):
            logger.warning(f"Plan failed Caring Invariants Firewall! Fallback triggered. Rec: {rec}")
            # Fallback to safe state: Ask user
            plan.actions = [Action(type="REQUEST_GUIDANCE", reversible=True)]
            plan.reasoning = ["Validation failed safety constraints. Requesting human guidance."]
            plan.ask_confirmation = True
            
        return plan

    def _apply_emergency_overrides(self, plan: ActionPlan, context: AgenticContext):
        """Phase 12 explicit lockouts."""
        if context.world_state.risk_weekly >= 0.75: # CRITICAL
            plan.actions = [Action(type="EMERGENCY_LOCKOUT", reversible=True)]
            plan.reasoning = ["CRITICAL weather risk. All automated watering suspended."]
            plan.add_alternatives = False
            
    def _apply_instincts(self, plan: ActionPlan, context: AgenticContext):
        """Agricultural heuristics specific to Puerto Rico."""
        # Instinct: Storm Prep
        if context.agent_state == AgentState.STORM_PREP:
            plan.reasoning.append("Instinct: Imminent storm. Avoid any transplanting and secure containers.")
            if any(a.type == "WATER" for a in plan.actions):
                plan.reasoning.append("Instinct: Reducing water to prevent root rot during incoming rain.")
                for a in plan.actions:
                    if a.type == "WATER":
                        a.multiplier *= 0.5 # Halve water ahead of storm
                        
        # Instinct: Flash Drought + Long Days
        if context.world_state.drought_active and context.world_state.temperature > 32.0:
            plan.reasoning.append("Instinct: High evaporation risk. Switching to split watering schedule.")
            for a in plan.actions:
                if a.type == "WATER":
                    a.split = True
                    
        # Instinct: Past Mistake Correction
        if context.past_mistakes:
            plan.reasoning.append("Instinct: Past failure noted in similar conditions. Adjusting behavior.")
            for mistake in context.past_mistakes:
                # Apply 3-strike autocorrects if they are ACTIVE or PERMANENT
                if mistake.get("status") in ["ACTIVE", "PERMANENT"] and mistake.get("correction_type") == "MULTIPLIER":
                    adj = mistake.get("correction_adjustment", 1.0)
                    if adj != 1.0:
                        plan.reasoning.append(f"Autocorrect: Applying {adj}x multiplier to WATER action due to past mistake.")
                        for a in plan.actions:
                            if a.type == "WATER":
                                a.multiplier *= adj

    def _apply_care_level(self, plan: ActionPlan, context: AgenticContext):
        """Modifies how the plan is delivered based on empathy state."""
        level = context.care_level
        state = context.agent_state
        
        if level >= 2:
            plan.break_into_steps = True
            plan.ask_confirmation = True
            
        if level >= 3 or state == AgentState.STORM_PREP:
            plan.prefer_reversible = True
            plan.add_alternatives = True
            
        if state == AgentState.SUPPORTIVE:
            plan.extra_explanation = True
            plan.gentler_tone = True
            
        if state == AgentState.RECOVERY:
            plan.gradual_normalization = True
            plan.monitoring_window_days = 3
