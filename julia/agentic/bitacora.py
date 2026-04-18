import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

logger = logging.getLogger("julia.agentic.bitacora")

class Bitacora:
    """
    Julia's decision journal — every action, every reason, every outcome.
    Works as the transparent audit trail of the agentic layer.
    """
    
    def __init__(self, db, log_path: str = "julia/logs/bitacora.jsonl"):
        self.db = db
        self.log_path = log_path
    
    def log(self, **kwargs):
        """Log a complete decision record to both SQLite and human-readable JSONL."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_state": kwargs.get("state", "NORMAL"),
            "care_level": kwargs.get("care_level", 0),
            "risk_probability": kwargs.get("risk_probability", 0.0),
            "risk_category": kwargs.get("risk_category", "LOW"),
            "care_triggers": kwargs.get("care_triggers", []),
            "recommendation": kwargs.get("recommendation", ""),
            "why": kwargs.get("why", []),
            "monitor_signal": kwargs.get("monitor_signal", ""),
            "actions": kwargs.get("actions", []),
            "confidence": kwargs.get("confidence", ""),
            "enso_phase": kwargs.get("enso_phase", ""),
            "corrections_applied": kwargs.get("corrections_applied", []),
        }
        
        # 1. Save to SQLite
        if self.db:
            try:
                self._insert_bitacora(entry)
            except Exception as e:
                logger.error(f"Failed to log to SQLite Bitacora: {e}")
        
        # 2. Save to readable log
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to JSONL Bitacora: {e}")
            
    def log_learning(self, event_type: str, description: str, mistake_id: int = None):
        """Log a learning event."""
        if self.db:
            with self.db._get_conn() as conn:
                conn.execute(
                    """INSERT INTO learning_events 
                       (event_type, description, related_mistake_id) 
                       VALUES (?, ?, ?)""",
                    (event_type, description, mistake_id)
                )
                conn.commit()
        logger.info(f"LEARNING [{event_type}]: {description}")

    def query_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the most recent bitacora entries so Julia can read her own history.
        This is critical for context building in Phase 18.
        """
        if not self.db:
            return []
            
        with self.db._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM bitacora
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            
            results = []
            for row in rows:
                r_dict = dict(row)
                # Parse the JSON fields back into objects
                for field in ['care_triggers', 'reasoning', 'actions', 'corrections_applied']:
                    if r_dict.get(field):
                        try:
                            r_dict[field] = json.loads(r_dict[field])
                        except Exception:
                            r_dict[field] = []
                results.append(r_dict)
            return results

    def _insert_bitacora(self, entry: Dict[str, Any]):
        """Internal helper to write the dict to SQLite."""
        with self.db._get_conn() as conn:
            conn.execute(
                """INSERT INTO bitacora
                   (timestamp, agent_state, care_level, risk_probability, risk_category, 
                    care_triggers, recommendation, reasoning, monitor_signal, actions, 
                    confidence, enso_phase, corrections_applied)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry["timestamp"],
                    str(entry["agent_state"]),
                    entry["care_level"],
                    entry["risk_probability"],
                    str(entry["risk_category"]),
                    json.dumps(entry["care_triggers"]),
                    str(entry["recommendation"]),
                    json.dumps(entry["why"]),
                    str(entry["monitor_signal"]),
                    json.dumps(entry["actions"]),
                    str(entry["confidence"]),
                    str(entry["enso_phase"]),
                    json.dumps(entry["corrections_applied"]),
                )
            )
            conn.commit()
