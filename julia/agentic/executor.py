import logging
from typing import Optional
from .state import AgenticContext
from .invariants import ActionPlan
from julia.core.config import JuliaConfig

logger = logging.getLogger("julia.agentic.executor")

class AgenticExecutor:
    """
    ACT phase of the OODA-L loop.
    Executes an ActionPlan. Enforces hardware guardrails and the Sandbox advisory mode.
    """
    
    def __init__(self, bitacora, config: Optional[JuliaConfig] = None):
        self.bitacora = bitacora
        self.config = config or JuliaConfig()
        
        # Sandbox Mode: If False, Julia only advises and logs. She physically CANNOT water.
        self.autonomous_mode = False 
        
        # Absolute Hardware Safety Limits (Phase 14 Caps)
        self.MAX_WATER_ML_PER_CYCLE = 500
        
    def execute(self, plan: ActionPlan, context: AgenticContext):
        """
        Processes the plan, applies hardware guardrails, logs to Bitacora, 
        and (if autonomous) executes hardware functions.
        """
        # 1. Apply hardware guardrails to all proposed actions
        self._apply_hardware_guardrails(plan)
        
        # 2. Build the final record meant for the user and long-term memory
        record = self._build_record(plan, context)
        
        # 3. Execution routing
        if not self.autonomous_mode:
            # Sandbox: Julia SUGGESTS, human DECIDES.
            logger.info("Sandbox Mode: Action Plan generated but hardware execution disabled.")
            if plan.ask_confirmation:
                logger.info("-> This plan would have requested human confirmation anyway.")
        else:
            # Autonomous: Julia acts, unless confirmation is explicitly required by Care Invariants
            if plan.ask_confirmation:
                logger.info("Autonomous Mode: Pausing for required human confirmation.")
                # In a real async system, this would trigger a push notification and wait
            else:
                logger.info("Autonomous Mode: Executing Action Plan on hardware.")
                self._trigger_hardware(plan)
                
        # 4. Log everything to the Bitacora so the Learner can evaluate it later
        self.bitacora.log(**record)
        return record
        
    def _apply_hardware_guardrails(self, plan: ActionPlan):
        """Absolute physical limits that no ML or Instinct can override."""
        for action in plan.actions:
            if action.type == "WATER":
                # Ensure the adjusted multiplier translates to safe ML amounts
                # Note: In a real system, amount_ml is calculated earlier but 
                # we enforce the max cap here right before hardware execution.
                if action.amount_ml > self.MAX_WATER_ML_PER_CYCLE:
                    logger.warning(f"Hardware Guardrail: Capping water amount from {action.amount_ml}ml to {self.MAX_WATER_ML_PER_CYCLE}ml")
                    action.amount_ml = self.MAX_WATER_ML_PER_CYCLE
                    plan.reasoning.append("Hardware Guardrail: Reduced water to maximum allowed per cycle (500ml).")
                    
    def _build_record(self, plan: ActionPlan, context: AgenticContext) -> dict:
        """Translates the plan and context into the schema expected by Bitacora."""
        return {
            "state": getattr(context.agent_state, "value", str(context.agent_state)),
            "care_level": context.care_level,
            "risk_probability": context.world_state.risk_weekly,
            "risk_category": "CRITICAL" if context.world_state.risk_weekly >= 0.75 else "HIGH" if context.world_state.risk_weekly >= 0.5 else "LOW",
            "care_triggers": context.detected_patterns,
            "recommendation": plan.primary_recommendation(),
            "why": plan.reasoning,
            # We assume the invariant check ensured proper monitor_signal logic before this phase
            "monitor_signal": "Monitor closely for 24h", 
            "actions": [a.to_dict() for a in plan.actions],
            "confidence": context.confidence,
            "enso_phase": context.world_state.enso_phase,
            "corrections_applied": [r for r in plan.reasoning if "Autocorrect" in r]
        }
        
    def _trigger_hardware(self, plan: ActionPlan):
        """Mock hardware execution."""
        for action in plan.actions:
            if action.type == "WATER":
                logger.info(f"HARDWARE: Pumping {action.amount_ml}ml")
            elif action.type == "ALERT_USER":
                logger.info("HARDWARE: Sending push notification")
