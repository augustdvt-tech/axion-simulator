"""
Axion AI - Operator Policies
============================

In a real plant deployment, decisions on each recommendation come from a human
operator. To validate the consensus loop end-to-end without a human in the
loop, we model the operator as an OperatorPolicy: an object that takes a
recommendation and returns a Decision.

This separation has two benefits:
1. We can test the consensus pipeline reproducibly with a deterministic policy.
2. When real operators come online, only the policy changes — the rest of the
   consensus machinery is identical. This is the same pattern used in
   reinforcement learning environments.

Policies provided:

- RealisticOperator: probabilistic responses keyed to urgency and confidence.
  - Critical / High: usually accept (operator trusts urgent diagnoses)
  - Medium: accept most, modify some, reject few
  - Low: reject many (operator doesn't have time for low-priority items)
  Modifications scale down the proposed adjustment slightly (operators are
  often more conservative than the recommendation engine).

- ConservativeOperator: rejects everything except critical events.
- AggressiveOperator: accepts everything.

Custom policies can be implemented by subclassing OperatorPolicy.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd

from recommendations import Recommendation, Action, Urgency, ActionType
from .models import Decision, DecisionStatus


class OperatorPolicy(ABC):
    """Abstract base for any operator response policy."""

    operator_id: str = "operator"

    @abstractmethod
    def decide(self, rec: Recommendation, timestamp: pd.Timestamp) -> Decision: ...


# =============================================================================
# Realistic operator
# =============================================================================

class RealisticOperator(OperatorPolicy):
    """
    Probabilistic policy that models a competent shift operator.

    Acceptance probability scales with urgency × confidence. Modification
    is more likely on MEDIUM/LOW recommendations where the operator has
    time to second-guess the magnitude. Rejection is most common on LOW
    items the operator considers noise.

    Modifications dampen the proposed adjustment by a configurable factor
    (default 0.7) — this captures the real-world tendency for operators to
    take a smaller step than the engine suggests, observe, and then move
    again if needed.
    """

    operator_id = "operator_sim"

    def __init__(
        self,
        seed: int = 42,
        modification_dampening: float = 0.7,
        accept_critical_prob: float = 0.95,
        accept_high_prob: float = 0.85,
        accept_medium_prob: float = 0.65,
        accept_low_prob: float = 0.30,
        modify_when_not_accepted_prob: float = 0.4,
    ):
        self.rng = np.random.default_rng(seed)
        self.modification_dampening = modification_dampening
        self.p_accept = {
            Urgency.CRITICAL: accept_critical_prob,
            Urgency.HIGH:     accept_high_prob,
            Urgency.MEDIUM:   accept_medium_prob,
            Urgency.LOW:      accept_low_prob,
        }
        self.p_modify_else = modify_when_not_accepted_prob

    def decide(self, rec: Recommendation, timestamp: pd.Timestamp) -> Decision:
        # Acceptance probability boosted by confidence
        p = self.p_accept[rec.urgency]
        p_effective = min(0.99, p * (0.5 + rec.confidence))   # confidence multiplier
        u = float(self.rng.random())

        if u < p_effective:
            # ACCEPT as-is
            return Decision(
                id=Decision.new_id(),
                recommendation_id=rec.id,
                timestamp=timestamp,
                operator_id=self.operator_id,
                status=DecisionStatus.ACCEPTED,
                actual_action=rec.action,
                justification="Approved by operator",
                rec_snapshot=rec.to_dict(),
            )

        # Not accepted: modify or reject?
        u2 = float(self.rng.random())
        if u2 < self.p_modify_else and rec.action.is_automated:
            # MODIFY: dampen the proposed adjustment
            new_action = self._dampen_action(rec.action)
            return Decision(
                id=Decision.new_id(),
                recommendation_id=rec.id,
                timestamp=timestamp,
                operator_id=self.operator_id,
                status=DecisionStatus.MODIFIED,
                actual_action=new_action,
                justification=(
                    f"Approved with smaller adjustment "
                    f"(dampened to {self.modification_dampening:.0%} of suggested)."
                ),
                rec_snapshot=rec.to_dict(),
            )

        # REJECT
        return Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            timestamp=timestamp,
            operator_id=self.operator_id,
            status=DecisionStatus.REJECTED,
            actual_action=None,
            justification=self._reject_reason(rec),
            rec_snapshot=rec.to_dict(),
        )

    def _dampen_action(self, original: Action) -> Action:
        """Return a copy of the action with the adjustment dampened."""
        if not original.is_automated or original.adjustment is None:
            return original
        new_adj = original.adjustment * self.modification_dampening
        new_proposed = (original.current_value or 0.0) + new_adj
        return Action(
            type=original.type,
            description=original.description + " [reduced by operator]",
            target_variable=original.target_variable,
            current_value=original.current_value,
            proposed_value=new_proposed,
            adjustment=new_adj,
            units=original.units,
        )

    def _reject_reason(self, rec: Recommendation) -> str:
        if rec.urgency == Urgency.LOW:
            return "Deferred — low priority, no immediate action warranted"
        return "Rejected — operator disagrees with diagnosis or proposed action"


# =============================================================================
# Conservative operator (only acts on critical events)
# =============================================================================

class ConservativeOperator(OperatorPolicy):
    operator_id = "operator_conservative"

    def decide(self, rec: Recommendation, timestamp: pd.Timestamp) -> Decision:
        if rec.urgency == Urgency.CRITICAL:
            status = DecisionStatus.ACCEPTED
            action = rec.action
            justification = "Approved (critical urgency)"
        else:
            status = DecisionStatus.REJECTED
            action = None
            justification = "Rejected — only acts on critical events"
        return Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            timestamp=timestamp,
            operator_id=self.operator_id,
            status=status,
            actual_action=action,
            justification=justification,
            rec_snapshot=rec.to_dict(),
        )


# =============================================================================
# Aggressive operator (accepts everything as-is)
# =============================================================================

class AggressiveOperator(OperatorPolicy):
    operator_id = "operator_aggressive"

    def decide(self, rec: Recommendation, timestamp: pd.Timestamp) -> Decision:
        return Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            timestamp=timestamp,
            operator_id=self.operator_id,
            status=DecisionStatus.ACCEPTED,
            actual_action=rec.action,
            justification="Auto-approved",
            rec_snapshot=rec.to_dict(),
        )
