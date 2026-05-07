"""
Axion AI - Consensus Controller
===============================

Orchestrates the human-machine consensus loop across three operating modes
(see Architecture Document section 4.5):

    1. ASESOR_PURO  (Advisor) — Axion AI recommends, operator decides and
       executes manually. The system never writes setpoints. Maximum safety;
       used during commissioning, on critical processes, and whenever
       confidence in the models is still being established.

    2. SEMI_AUTONOMOUS — Axion AI prepares the action and sends it to the
       operator for one-click approval. On approval the system executes
       automatically. Operator remains in the loop but the friction is
       minimized.

    3. AUTONOMOUS_SUPERVISED — Axion AI executes within pre-approved limits
       and notifies post-execution. Used for low-risk, well-validated
       optimizations (e.g. small reflux adjustments). Anything outside
       pre-approved limits escalates back to semi-autonomous mode.

Safety invariants (independent of operating mode):

    HARD_LIMITS — Axion AI never recommends or executes an action that
    would push a variable outside its engineering safety limits. These
    limits are separate from operational limits; they're set by an
    administrator and represent physical/regulatory constraints.

    MIN_ACTION_SIZE — actions below a minimum magnitude are filtered out
    to avoid noisy micro-adjustments.

    ROLLBACK_POLICY — in semi-autonomous and autonomous modes, if an
    executed action produces an outcome that diverges significantly from
    the predicted impact (configurable threshold), the system reverts the
    action and alerts the operator.

The ConsensusController is stateless beyond configuration — every method
takes the relevant inputs as arguments. State (decisions, executions,
outcomes) lives in the DecisionLog.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
import pandas as pd

from recommendations import Recommendation, Action, ActionType, Urgency
from .models import (
    Decision, DecisionStatus, Execution, ExecutionStatus, Outcome,
)
from .operator import OperatorPolicy
from .outcome import OutcomeTracker, PerformanceTracker


class OperatingMode(str, Enum):
    ADVISOR = "advisor"                   # operator does everything manually
    SEMI_AUTONOMOUS = "semi_autonomous"   # operator approves, system executes
    AUTONOMOUS_SUPERVISED = "autonomous_supervised"  # system executes within limits


# =============================================================================
# Safety configuration
# =============================================================================

@dataclass
class SafetyLimits:
    """
    Hard engineering safety limits per variable. Different from operational
    limits (which are 'where the operator wants to operate'); these are
    'where the process is physically safe to operate'.
    """
    limits: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def is_within_limits(self, variable: str, value: float) -> bool:
        if variable not in self.limits:
            return True
        lim = self.limits[variable]
        if "min" in lim and value < lim["min"]:
            return False
        if "max" in lim and value > lim["max"]:
            return False
        return True


# Default safety limits for the pilot process. These bracket each variable's
# normal operational range with a safety margin. Source: same engineering
# constraints used in the simulator's process design.
PILOT_SAFETY_LIMITS = SafetyLimits(limits={
    "cstr.F_cool":     {"min": 0.05, "max": 0.80},
    "cstr.F_feed":     {"min": 1.5,  "max": 3.0},
    "column.RR":       {"min": 2.5,  "max": 8.0},
})


# =============================================================================
# Auto-approval rules for AUTONOMOUS_SUPERVISED mode
# =============================================================================

@dataclass
class AutonomyRules:
    """
    Defines what kinds of actions can be executed without operator approval
    in AUTONOMOUS_SUPERVISED mode. Anything not matching here falls back
    to operator approval (semi-autonomous behavior).

    The default policy below is intentionally conservative: only LOW-urgency
    setpoint adjustments below a small magnitude. This is the right starting
    posture — autonomy is earned by demonstrated track record (via the
    PerformanceTracker), not granted by default.
    """
    # Allowed urgency levels for autonomous execution
    allowed_urgencies: Tuple[Urgency, ...] = (Urgency.LOW,)
    # Allowed action types
    allowed_action_types: Tuple[ActionType, ...] = (ActionType.ADJUST_SETPOINT,)
    # Maximum |adjustment| / |current_value| allowed to execute autonomously
    max_relative_adjustment: float = 0.10
    # Minimum confidence required for autonomous execution
    min_confidence: float = 0.85
    # Tags this rules applies to (autonomous moves only on these variables)
    allowed_target_variables: Tuple[str, ...] = ("column.RR",)


# =============================================================================
# Decision Log (in-memory store of all interactions)
# =============================================================================

class DecisionLog:
    """In-memory store of decisions, executions, and outcomes."""
    def __init__(self):
        self.recommendations: List[Recommendation] = []
        self.decisions: List[Decision] = []
        self.executions: List[Execution] = []
        self.outcomes: List[Outcome] = []

        # Cross-reference indexes
        self._rec_by_id: Dict[str, Recommendation] = {}
        self._dec_by_rec: Dict[str, Decision] = {}
        self._exe_by_dec: Dict[str, Execution] = {}

    def add_recommendation(self, rec: Recommendation):
        self.recommendations.append(rec)
        self._rec_by_id[rec.id] = rec

    def add_decision(self, dec: Decision):
        self.decisions.append(dec)
        self._dec_by_rec[dec.recommendation_id] = dec

    def add_execution(self, exe: Execution):
        self.executions.append(exe)
        self._exe_by_dec[exe.decision_id] = exe

    def add_outcome(self, outcome: Outcome):
        self.outcomes.append(outcome)

    def get_recommendation(self, rec_id: str) -> Optional[Recommendation]:
        return self._rec_by_id.get(rec_id)

    def summary(self) -> Dict[str, int]:
        return {
            "recommendations": len(self.recommendations),
            "decisions":       len(self.decisions),
            "accepted":        sum(1 for d in self.decisions if d.status == DecisionStatus.ACCEPTED),
            "modified":        sum(1 for d in self.decisions if d.status == DecisionStatus.MODIFIED),
            "rejected":        sum(1 for d in self.decisions if d.status == DecisionStatus.REJECTED),
            "auto_executed":   sum(1 for d in self.decisions if d.status == DecisionStatus.AUTO_EXECUTED),
            "executions":      len(self.executions),
            "outcomes":        len(self.outcomes),
        }


# =============================================================================
# ConsensusController
# =============================================================================

class ConsensusController:
    """
    Processes recommendations through the consensus loop and produces a
    DecisionLog with full traceability.

    Usage:
        controller = ConsensusController(
            mode=OperatingMode.SEMI_AUTONOMOUS,
            operator=RealisticOperator(),
            safety_limits=PILOT_SAFETY_LIMITS,
        )
        log = controller.process(
            recommendations=recs,
            process_data=df,
        )
    """

    def __init__(
        self,
        mode: OperatingMode = OperatingMode.SEMI_AUTONOMOUS,
        operator: Optional[OperatorPolicy] = None,
        safety_limits: Optional[SafetyLimits] = None,
        autonomy_rules: Optional[AutonomyRules] = None,
        outcome_tracker: Optional[OutcomeTracker] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        min_action_size: float = 1e-3,
    ):
        self.mode = mode
        self.operator = operator
        self.safety_limits = safety_limits or PILOT_SAFETY_LIMITS
        self.autonomy_rules = autonomy_rules or AutonomyRules()
        self.outcome_tracker = outcome_tracker or OutcomeTracker()
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.min_action_size = min_action_size

    # ------ main pipeline ------

    def process(
        self,
        recommendations: List[Recommendation],
        process_data: pd.DataFrame,
    ) -> DecisionLog:
        """Process all recommendations through consensus → execution → outcomes."""
        log = DecisionLog()

        for rec in recommendations:
            log.add_recommendation(rec)
            self.performance_tracker.record_recommendation(rec)

            # 1) Safety gate: never let an unsafe action through
            if not self._safety_check(rec):
                # Auto-reject for safety reasons
                dec = Decision(
                    id=Decision.new_id(),
                    recommendation_id=rec.id,
                    timestamp=rec.timestamp,
                    operator_id="system_safety",
                    status=DecisionStatus.REJECTED,
                    actual_action=None,
                    justification="Auto-rejected by safety gate: action would exceed engineering limits",
                    rec_snapshot=rec.to_dict(),
                )
                log.add_decision(dec)
                self.performance_tracker.record_decision(rec, dec)
                continue

            # 2) Filter trivially small actions
            if self._is_too_small(rec.action):
                dec = Decision(
                    id=Decision.new_id(),
                    recommendation_id=rec.id,
                    timestamp=rec.timestamp,
                    operator_id="system_filter",
                    status=DecisionStatus.REJECTED,
                    actual_action=None,
                    justification="Suppressed: action magnitude below minimum",
                    rec_snapshot=rec.to_dict(),
                )
                log.add_decision(dec)
                self.performance_tracker.record_decision(rec, dec)
                continue

            # 3) Mode-dependent decision flow
            if self.mode == OperatingMode.AUTONOMOUS_SUPERVISED and self._qualifies_for_autonomy(rec):
                dec = self._auto_decide(rec)
            elif self.operator is not None:
                # Both ADVISOR and SEMI_AUTONOMOUS modes route through the operator
                dec = self.operator.decide(rec, rec.timestamp)
            else:
                # No operator policy attached — recommendations remain pending
                dec = Decision(
                    id=Decision.new_id(),
                    recommendation_id=rec.id,
                    timestamp=rec.timestamp,
                    operator_id="none",
                    status=DecisionStatus.PENDING,
                    actual_action=None,
                    justification="No operator policy attached",
                    rec_snapshot=rec.to_dict(),
                )

            log.add_decision(dec)
            self.performance_tracker.record_decision(rec, dec)

            # 4) Execute (or not, depending on mode and decision)
            execution = self._execute_if_appropriate(rec, dec)
            if execution is not None:
                log.add_execution(execution)

                # 5) Measure outcome (in our simulation we have all data
                # available; in production this would be scheduled async)
                outcome = self.outcome_tracker.measure(rec, dec, execution, process_data)
                if outcome is not None:
                    log.add_outcome(outcome)
                    self.performance_tracker.record_outcome(rec, outcome)

        return log

    # ------ helpers ------

    def _safety_check(self, rec: Recommendation) -> bool:
        """Return True if the recommendation's action is within safety limits."""
        a = rec.action
        if a.target_variable is None or a.proposed_value is None:
            return True   # non-setpoint action; nothing to check
        return self.safety_limits.is_within_limits(a.target_variable, a.proposed_value)

    def _is_too_small(self, action: Action) -> bool:
        """Return True if this action is below the minimum magnitude."""
        if not action.is_automated:
            return False
        if action.adjustment is None:
            return False
        return abs(action.adjustment) < self.min_action_size

    def _qualifies_for_autonomy(self, rec: Recommendation) -> bool:
        """Check whether a recommendation matches the autonomy rules."""
        rules = self.autonomy_rules
        if rec.urgency not in rules.allowed_urgencies:
            return False
        if rec.action.type not in rules.allowed_action_types:
            return False
        if rec.action.target_variable not in rules.allowed_target_variables:
            return False
        if rec.confidence < rules.min_confidence:
            return False
        # Magnitude check
        if (rec.action.current_value is not None
            and rec.action.adjustment is not None
            and abs(rec.action.current_value) > 1e-9):
            rel = abs(rec.action.adjustment / rec.action.current_value)
            if rel > rules.max_relative_adjustment:
                return False
        return True

    def _auto_decide(self, rec: Recommendation) -> Decision:
        """Build a Decision for an autonomous execution."""
        return Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            timestamp=rec.timestamp,
            operator_id="system_autonomous",
            status=DecisionStatus.AUTO_EXECUTED,
            actual_action=rec.action,
            justification="Auto-executed (within autonomy rules)",
            rec_snapshot=rec.to_dict(),
        )

    def _execute_if_appropriate(
        self, rec: Recommendation, decision: Decision
    ) -> Optional[Execution]:
        """Decide whether to execute and produce an Execution record."""
        # Only ACCEPTED / MODIFIED / AUTO_EXECUTED decisions execute
        if decision.status not in (
            DecisionStatus.ACCEPTED,
            DecisionStatus.MODIFIED,
            DecisionStatus.AUTO_EXECUTED,
        ):
            return None

        action = decision.actual_action
        if action is None:
            return None

        # Non-setpoint actions execute as NOT_REQUIRED (operator action only)
        if not action.is_automated:
            return Execution(
                id=Execution.new_id(),
                decision_id=decision.id,
                timestamp=decision.timestamp,
                status=ExecutionStatus.NOT_REQUIRED,
                executor=decision.operator_id,
            )

        # In ADVISOR mode, the operator executes manually — we still record
        # the execution but mark it NOT_REQUIRED to distinguish from
        # system-driven setpoint writes
        if self.mode == OperatingMode.ADVISOR:
            return Execution(
                id=Execution.new_id(),
                decision_id=decision.id,
                timestamp=decision.timestamp,
                status=ExecutionStatus.NOT_REQUIRED,
                executor="operator_manual",
            )

        # SEMI_AUTONOMOUS / AUTONOMOUS: simulated setpoint write succeeds.
        # Real production would call out to an OPC-UA writer here and
        # capture errors. For the MVP we always succeed.
        executor = (
            "system_autonomous"
            if decision.status == DecisionStatus.AUTO_EXECUTED
            else "system_semi_auto"
        )
        return Execution(
            id=Execution.new_id(),
            decision_id=decision.id,
            timestamp=decision.timestamp,
            status=ExecutionStatus.SUCCESS,
            executor=executor,
        )
