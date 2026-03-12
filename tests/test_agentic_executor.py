import pytest
from julia.agentic.executor import AgenticExecutor
from julia.agentic.invariants import ActionPlan, Action
from julia.agentic.state import AgenticContext, WorldState
from julia.agentic.caring import AgentState

class MockBitacora:
    def __init__(self):
        self.logs = []
        
    def log(self, **kwargs):
        self.logs.append(kwargs)
        
@pytest.fixture
def executor():
    return AgenticExecutor(bitacora=MockBitacora())

def test_executor_hardware_guardrails(executor):
    # Attempt to execute a plan with a massive amount of water (1000ml)
    plan = ActionPlan(
        actions=[Action(type="WATER", amount_ml=1000, reversible=False)],
        reasoning=["Testing massive water payload."],
        confidence="HIGH"
    )
    
    ctx = AgenticContext(world_state=WorldState(), care_level=1, agent_state=AgentState.NORMAL)
    
    record = executor.execute(plan, ctx)
    
    # The guardrail should have capped it at 500ml
    assert plan.actions[0].amount_ml == 500
    assert any("Hardware Guardrail" in r for r in plan.reasoning)
    
    # Record logged properly
    assert len(executor.bitacora.logs) == 1
    assert executor.bitacora.logs[0]["actions"][0]["amount_ml"] == 500

def test_executor_sandbox_mode(executor):
    plan = ActionPlan(
        actions=[Action(type="WATER", amount_ml=200, reversible=False)],
        reasoning=["Normal watering."],
        confidence="HIGH"
    )
    ctx = AgenticContext(world_state=WorldState(), care_level=1, agent_state=AgentState.NORMAL)
    
    executor.autonomous_mode = False
    
    # Execute should return without triggering hardware (we test this by observing no side-effects or exceptions)
    record = executor.execute(plan, ctx)
    
    # It still logged the INTENT
    assert len(executor.bitacora.logs) == 1
    assert executor.bitacora.logs[0]["actions"][0]["amount_ml"] == 200
