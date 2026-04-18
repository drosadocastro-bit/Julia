from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class Action:
    type: str # WATER, ALERT_USER, ADJUST_SCHEDULE, LOG_OBSERVATION, REQUEST_GUIDANCE, DO_NOTHING, EMERGENCY
    reversible: bool
    multiplier: float = 1.0
    amount_ml: float = 0.0
    split: bool = False
    
    def to_dict(self):
        return {
            "type": self.type,
            "reversible": self.reversible,
            "multiplier": self.multiplier,
            "amount_ml": self.amount_ml,
            "split": self.split
        }

@dataclass
class Recommendation:
    action: Action
    reasoning: Optional[str] = None
    monitor_signal: Optional[str] = None
    context_risk: float = 0.0
    uncertainty: float = 0.0
    asks_clarification: bool = False
    
    def exceeds_user_constraints(self) -> bool:
        """
        Placeholder for checking against user constraints (time, water limits).
        Assume False for now.
        """
        return False

@dataclass
class ActionPlan:
    actions: List[Action]
    reasoning: List[str]
    confidence: str
    break_into_steps: bool = False
    ask_confirmation: bool = False
    prefer_reversible: bool = False
    add_alternatives: bool = False
    extra_explanation: bool = False
    gentler_tone: bool = False
    gradual_normalization: bool = False
    monitoring_window_days: int = 0
    
    def primary_recommendation(self) -> str:
        """Helper to get a human-readable recommendation."""
        actions_str = ", ".join(a.type for a in self.actions)
        return f"Plan: {actions_str} (Break steps: {self.break_into_steps})"

class CaringInvariants:
    """
    5 unbreakable rules Julia NEVER breaks. Checked on every output.
    """
    
    def validate(self, recommendation: Recommendation) -> bool:
        return all([
            self.has_why(recommendation),
            self.has_monitor_signal(recommendation),
            self.respects_reversibility(recommendation),
            self.handles_uncertainty(recommendation),
            self.respects_user_constraints(recommendation)
        ])
    
    def has_why(self, rec: Recommendation) -> bool:
        """INVARIANT 1: Always explain WHY behind recommendation."""
        return rec.reasoning is not None and len(rec.reasoning) > 0
    
    def has_monitor_signal(self, rec: Recommendation) -> bool:
        """INVARIANT 2: Always offer 1 observable signal to monitor."""
        return rec.monitor_signal is not None and len(rec.monitor_signal) > 0
    
    def respects_reversibility(self, rec: Recommendation) -> bool:
        """INVARIANT 3: Prefer reversible action when risk > 0.5."""
        if rec.context_risk > 0.5:
            return rec.action.reversible
        return True  # No constraint when risk is low
    
    def handles_uncertainty(self, rec: Recommendation) -> bool:
        """INVARIANT 4: Ask clarification if uncertainty > 0.6."""
        if rec.uncertainty > 0.6:
            return rec.asks_clarification
        return True
    
    def respects_user_constraints(self, rec: Recommendation) -> bool:
        """INVARIANT 5: Respect user constraints (time, water, tools)."""
        return not rec.exceeds_user_constraints()
