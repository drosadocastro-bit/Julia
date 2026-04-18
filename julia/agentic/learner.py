import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from .memory import MistakeMemory

logger = logging.getLogger("julia.agentic.learner")

class AgenticLearner:
    """
    LEARN phase of the OODA-L loop.
    Evaluates past decisions against actual outcomes. Issues strikes and autocorrects.
    """
    
    def __init__(self, db, bitacora):
        self.db = db
        self.bitacora = bitacora
        self.memory = MistakeMemory(db)
        
    def evaluate_outcomes(self, hours_back: int = 24, evaluation_window_hours: int = 4):
        """
        Runs periodically (e.g. daily).
        Checks all watering decisions from `hours_back` ago.
        Looks at the sensor data exactly `evaluation_window_hours` AFTER the watering event.
        If the outcome deviated significantly from expectations, registers a mistake.
        """
        logger.info(f"Agentic Learner: Evaluating outcomes for the past {hours_back} hours.")
        
        # 1. Fetch recent actions
        recent_decisions = self.bitacora.query_recent(limit=50) # Assuming we can filter by timestamp in real DB
        
        now = datetime.now()
        for decision in recent_decisions:
            decision_time = datetime.fromisoformat(decision["timestamp"])
            
            # We only evaluate if enough time has passed to see the true outcome (e.g. 4 hours)
            if now - decision_time < timedelta(hours=evaluation_window_hours):
                continue
                
            # Filter for watering actions
            water_actions = [a for a in decision.get("actions", []) if a.get("type") == "WATER"]
            if not water_actions:
                continue
                
            # For each watering action, check the actual outcome in the database
            for action in water_actions:
                amount_ml = action.get("amount_ml", 0)
                # We need the sensor data exactly at `decision_time + evaluation_window`
                target_eval_time = decision_time + timedelta(hours=evaluation_window_hours)
                
                outcome = self._get_true_hardware_outcome(target_eval_time)
                
                # Check deviations
                deviation, error_type, recommended_patch = self._calculate_deviation(decision, outcome, amount_ml)
                
                if deviation:
                    self._register_mistake(decision, error_type, recommended_patch)
                    
    def _get_true_hardware_outcome(self, target_time: datetime) -> Dict[str, float]:
        """Query SQLite for exactly what happened at the target time."""
        if not self.db:
            return {"soil_moisture": 60.0} # Stub
            
        try:
            # E.g. fetch readings around target time
            res = self.db.get_sensor_trend("basil", hours=1) # Simplified for stub
            if res:
                return {"soil_moisture": res[-1].get("soil_moisture", 50.0)}
        except Exception as e:
            logger.error(f"Learner failed to query outcome: {e}")
            
        return {"soil_moisture": 50.0}
        
    def _calculate_deviation(self, decision: Dict, actual_outcome: Dict, water_amount_ml: float):
        """
        Compares expectation vs reality. We expect 500ml of water to raise moisture to e.g. 70%.
        If it only hit 40% (Undershoot), or hit 95% (Overshoot), we calculate a correction.
        """
        # Simplified heuristics for the architectural stub
        actual_moisture = actual_outcome.get("soil_moisture", 50.0)
        expected_moisture = 70.0 # Constant for now, normally based on plant profile max_moisture
        
        if actual_moisture < expected_moisture - 15:
            # Undershoot: Soil is still very dry after watering
            adj = 1.2 # Increase multiplier by 20%
            return True, "UNDERSHOOT", adj
            
        elif actual_moisture > expected_moisture + 15:
            # Overshoot: Soil is drowned
            adj = 0.8 # Decrease multiplier by 20%
            return True, "OVERSHOOT", adj
            
        return False, "NOMINAL", 1.0
        
    def _register_mistake(self, decision: Dict, error_type: str, correction_adjustment: float):
        """
        Logs the mistake to MistakeMemory and applies the 3-Strike rule.
        """
        # We need the original conditions_hash that led to this decision
        # We didn't store it explicitly in Bitacora schema originally, but we should calculate or store it.
        # For the architecture stub, we'll assume we can pass a mock hash or retrieve it.
        hash_id = decision.get("hash", "dummy_hash") 
        
        # Log to structural Mistake table
        mistake_id = self.memory.record_mistake(
            conditions_hash=hash_id,
            action_taken=str(decision.get("actions")),
            expected_outcome=f"Moisture target reached.",
            actual_outcome=f"Error: {error_type}",
            error_type=error_type,
            plant_id="UNKNOWN", # In fully integrated version, split by plant
            correction_type="MULTIPLIER",
            correction_adjustment=correction_adjustment
        )
        
        # Strike Logic
        self._apply_strike_rules(hash_id, error_type, mistake_id)
        
    def _apply_strike_rules(self, conditions_hash: str, error_type: str, mistake_id: int):
        """
        The 3-Strike Rule Mechanic.
        1 or 2 mistakes = ACTIVE correction.
        3 mistakes = PERMANENT threshold rewrite warning.
        """
        if not self.db:
            return
            
        # Count identical mistakes under this exact condition
        with self.db._get_conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as count FROM mistakes
                   WHERE conditions_hash = ? AND error_type = ?""",
                (conditions_hash, error_type)
            ).fetchone()
            count = row["count"]
            
            if count == 3:
                # STRIKE 3: Upgrade to PERMANENT
                logger.warning(f"STRIKE 3 for hash {conditions_hash}. Upgrading {error_type} correction to PERMANENT.")
                conn.execute(
                    """UPDATE mistakes SET status = 'PERMANENT' WHERE conditions_hash = ?""",
                    (conditions_hash,)
                )
                
                # Log the meta-learning event to Bitacora
                self.bitacora.log_learning("STRIKE_3_UPGRADE", f"Mistake escalated to PERMANENT mapping for hash {conditions_hash}", mistake_id)
            else:
                logger.info(f"Strike {count} for hash {conditions_hash}. Logged as ACTIVE correction.")
                
            conn.commit()
