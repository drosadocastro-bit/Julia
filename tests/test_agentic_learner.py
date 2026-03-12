import pytest
from julia.agentic.learner import AgenticLearner

class MockMistakeMemory:
    def record_mistake(self, conditions_hash, action_taken, expected_outcome, actual_outcome, error_type, plant_id, correction_type, correction_adj):
        return 1

class MockDB:
    def __init__(self):
        self.call_count = 0
        self.strike_count_to_return = 1
        self.status_updated_to = None
        
    def _get_conn(self):
        return self
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def execute(self, query, params=None):
        class MockCursor:
            def __init__(self, db_instance):
                self.db = db_instance
            def fetchone(self):
                return {"count": self.db.strike_count_to_return}
                
        if "UPDATE mistakes SET status" in query:
            self.status_updated_to = "PERMANENT"
            
        return MockCursor(self)
        
    def commit(self):
        pass

class MockBitacora2:
    def log_learning(self, event_type, details, mistake_id):
        pass

def test_learner_3_strike_rule():
    db = MockDB()
    bitacora = MockBitacora2()
    learner = AgenticLearner(db, bitacora)
    learner.memory = MockMistakeMemory()
    
    # Test Strike 1 (ACTIVE)
    db.strike_count_to_return = 1
    learner._apply_strike_rules("hash123", "UNDERSHOOT", 1)
    assert db.status_updated_to is None
    
    # Test Strike 3 (PERMANENT UPGRADE)
    db.strike_count_to_return = 3
    learner._apply_strike_rules("hash123", "UNDERSHOOT", 2)
    assert db.status_updated_to == "PERMANENT"
