from typing import List, Dict, Any, Optional

class MistakeMemory:
    """
    Retrieves and manages mistakes to prevent Julia from repeating them.
    Tied to the 'mistakes' table in SQLite.
    """
    
    def __init__(self, db):
        self.db = db
    
    def get_mistakes(self, similar_conditions: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent mistakes for a specific condition hash."""
        if not self.db:
            return []
            
        with self.db._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM mistakes
                   WHERE conditions_hash = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (similar_conditions, limit)
            ).fetchall()
            return [dict(r) for r in rows]
            
    def has_similar_failure(self, conditions_hash: str, days: int = 60) -> bool:
        """Check if there was a failure under these exact conditions recently."""
        if not self.db:
            return False
            
        with self.db._get_conn() as conn:
            # We assume datetime('now', '-60 days') logic
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM mistakes
                   WHERE conditions_hash = ?
                     AND timestamp >= datetime('now', ?)""",
                (conditions_hash, f"-{days} days")
            ).fetchone()
            return row["cnt"] > 0
            
    def get_corrections_for(self, conditions_hash: str) -> List[Dict[str, Any]]:
        """Retrieve active or permanent autocorrect instructions."""
        if not self.db:
            return []
            
        with self.db._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM mistakes
                   WHERE conditions_hash = ?
                     AND status IN ('ACTIVE', 'PERMANENT')
                     AND correction_type IS NOT NULL""",
                (conditions_hash,)
            ).fetchall()
            return [dict(r) for r in rows]
            
    def record_mistake(
        self,
        conditions_hash: str,
        action_taken: str,
        expected_outcome: str,
        actual_outcome: str,
        error_type: str,
        plant_id: Optional[str] = None,
        correction_type: Optional[str] = None,
        correction_param: Optional[str] = None,
        correction_adjustment: Optional[float] = None
    ) -> int:
        """Log a new mistake discovered by the learning engine."""
        if not self.db:
            return -1
            
        with self.db._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO mistakes
                   (conditions_hash, plant_id, action_taken, expected_outcome, actual_outcome, 
                    error_type, correction_type, correction_param, correction_adjustment)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (conditions_hash, plant_id, action_taken, expected_outcome, actual_outcome,
                 error_type, correction_type, correction_param, correction_adjustment)
            )
            conn.commit()
            return cursor.lastrowid
