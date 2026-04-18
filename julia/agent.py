import logging
import time
from typing import Optional, Dict, Any

from julia.core.config import JuliaConfig
from julia.core.decision_engine import JuliaDecisionEngine

from julia.agentic.bitacora import Bitacora
from julia.agentic.perception import PerceptionLayer
from julia.agentic.context_engine import ContextEngine
from julia.agentic.planner import AgenticPlanner
from julia.agentic.executor import AgenticExecutor
from julia.agentic.learner import AgenticLearner

logger = logging.getLogger("julia.agent")

class JuliaAgent:
    """
    Agentic Julia v2: Master Orchestrator.
    Wires together the four phases of the OODA-L loop: OBSERVE, ORIENT, DECIDE, ACT + LEARN.
    """
    
    def __init__(self, db, weather_service, climate_risk_engine, config: Optional[JuliaConfig] = None):
        logger.info("Initializing Agentic Julia v2 OODA-L Framework...")
        self.config = config or JuliaConfig()
        
        # Core Infrastructure
        self.db = db
        self.weather_service = weather_service
        self.climate_risk_engine = climate_risk_engine
        
        # Logging & Memory
        self.bitacora = Bitacora(db)
        
        # 1. OBSERVE
        self.perception = PerceptionLayer(db, weather_service, climate_risk_engine)
        
        # 2. ORIENT
        self.context_engine = ContextEngine(db)
        
        # 3. DECIDE
        base_decision_engine = JuliaDecisionEngine(self.config, db)
        self.planner = AgenticPlanner(decision_engine=base_decision_engine)
        
        # 4. ACT
        self.executor = AgenticExecutor(self.bitacora, self.config)
        
        # 5. LEARN
        self.learner = AgenticLearner(db, self.bitacora)

    @property
    def sandbox_mode(self) -> bool:
        """Returns True if Julia is operating in Sandbox (Advisory-only) mode."""
        return not self.executor.autonomous_mode
        
    @sandbox_mode.setter
    def sandbox_mode(self, value: bool):
        """Toggle Julia's Sandbox limitation."""
        self.executor.autonomous_mode = not value
        logger.info(f"Agentic Sandbox Mode set to: {value}")
        
    def tick(self, plant_id: str = "all") -> Dict[str, Any]:
        """
        Executes a single cycle of the OODA-L agentic loop.
        """
        logger.info(f"--- Starting Agentic Loop (Target: {plant_id}) ---")
        start_time = time.time()
        
        try:
            # 1. OBSERVE (Perception)
            logger.debug("Step 1/4: OBSERVE")
            world_state = self.perception.get_world_state()
            
            # 2. ORIENT (Context Engine)
            logger.debug("Step 2/4: ORIENT")
            agentic_context = self.context_engine.build_context(world_state)
            
            # 3. DECIDE (Planner + Instincts + Invariants)
            logger.debug("Step 3/4: DECIDE")
            action_plan = self.planner.plan(agentic_context)
            
            # 4. ACT (Executor + Guardrails)
            logger.debug("Step 4/4: ACT")
            execution_record = self.executor.execute(action_plan, agentic_context)
            
            duration = (time.time() - start_time) * 1000
            logger.info(f"--- Loop Completed in {duration:.2f}ms ---")
            
            return execution_record
            
        except Exception as e:
            logger.exception(f"CRITICAL ERROR in Agentic OODA Loop: {e}")
            # Failsafe execution log
            failsafe_record = {
                "state": "RECOVERY",
                "care_level": 3,
                "recommendation": "System panicked. Asking human.",
                "why": [f"Uncaught exception in tick(): {str(e)}"],
                "actions": [{"type": "REQUEST_GUIDANCE", "reversible": True}]
            }
            self.bitacora.log(**failsafe_record)
            return failsafe_record

    def run_daily_reflection(self):
        """
        The meta-learning cycle. Usually run asynchronously at night.
        Evaluates the results of executed actions and applies strikes/autocorrects.
        """
        logger.info("Running Daily Agentic Reflection...")
        try:
            self.learner.evaluate_outcomes(hours_back=24, evaluation_window_hours=4)
            logger.info("Daily Reflection Complete.")
        except Exception as e:
            logger.exception(f"Error during Daily Reflection: {e}")
