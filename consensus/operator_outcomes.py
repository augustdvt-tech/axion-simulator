"""
Axion AI — Operator outcome tracking
=====================================

Closes the loop on UI-recorded operator decisions:

    operator clicks "accept" via the UI
        ↓
    we record { rec_id → (status, justification, decision_ts) }
        ↓
    once the replay clock passes decision_ts + max_measurement_delay
        ↓
    we synthesize a Decision + Execution and feed both, plus the
    Recommendation and the process_data, to the existing OutcomeTracker
        ↓
    Outcome (with predicted vs actual values + quality score) is stored
    and surfaced via /api/outcomes/operator and the dashboard.

The simulator's `RealisticOperator` already follows the same flow for
synthetic decisions — this module does the same for UI-driven decisions
without having to refactor the core OutcomeTracker.

Design notes
------------
- Pure functions: no FastAPI imports, no global state. The server is the
  one that wires these helpers into its replay loop.
- Idempotent: `measure_pending` returns only outcomes that are *now*
  measurable AND haven't been measured yet (caller passes the set of
  already-measured rec ids).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from recommendations import Recommendation
from .models import (
    Decision, DecisionStatus, Execution, ExecutionStatus, Outcome,
)
from .outcome import OutcomeTracker


@dataclass
class OperatorOverride:
    """The state we keep in memory for each UI decision."""
    rec_id:        str
    status:        str             # "accepted" | "rejected" | "modified"
    justification: str
    decision_ts:   pd.Timestamp    # simulation time at which the operator clicked


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _max_measurement_delay(rec: Recommendation, default_delay_min: float = 30.0) -> float:
    """Largest expected measurement delay across the rec's expected_impact list."""
    if not rec.expected_impact:
        return default_delay_min
    return max(
        (imp.time_to_effect_minutes or default_delay_min)
        for imp in rec.expected_impact
    )


def is_outcome_measurable(
    override: OperatorOverride,
    rec: Recommendation,
    now_ts: pd.Timestamp,
    default_delay_min: float = 30.0,
) -> bool:
    """True iff the replay clock has advanced past decision_ts + max delay
    AND the decision was actionable (accepted / modified)."""
    if override.status not in ("accepted", "modified"):
        return False
    if not rec.expected_impact:
        return False
    delay = _max_measurement_delay(rec, default_delay_min)
    return now_ts >= override.decision_ts + pd.Timedelta(minutes=delay)


def synthesize_decision_and_execution(
    rec: Recommendation,
    override: OperatorOverride,
) -> tuple[Decision, Execution]:
    """Build the Decision + Execution objects that OutcomeTracker expects."""
    status_map = {
        "accepted": DecisionStatus.ACCEPTED,
        "rejected": DecisionStatus.REJECTED,
        "modified": DecisionStatus.MODIFIED,
    }
    decision = Decision(
        id=Decision.new_id(),
        recommendation_id=rec.id,
        timestamp=override.decision_ts,
        status=status_map.get(override.status, DecisionStatus.ACCEPTED),
        operator_id="operator_ui",
        justification=override.justification,
    )
    execution = Execution(
        id=Execution.new_id(),
        decision_id=decision.id,
        timestamp=override.decision_ts,
        status=ExecutionStatus.SUCCESS,
        executor="operator",
    )
    return decision, execution


def measure_one(
    override: OperatorOverride,
    rec: Recommendation,
    process_data: pd.DataFrame,
    tracker: Optional[OutcomeTracker] = None,
) -> Optional[Outcome]:
    """Synthesize Decision/Execution and run OutcomeTracker.measure."""
    tracker = tracker or OutcomeTracker()
    decision, execution = synthesize_decision_and_execution(rec, override)
    return tracker.measure(rec, decision, execution, process_data)


def measure_pending(
    overrides: Dict[str, OperatorOverride],
    recs_by_id: Dict[str, Recommendation],
    now_ts: pd.Timestamp,
    process_data: pd.DataFrame,
    already_measured: Set[str],
    tracker: Optional[OutcomeTracker] = None,
) -> List[Outcome]:
    """Return outcomes that are newly measurable. Skips already-measured rec ids
    and skips overrides whose recommendation can't be looked up."""
    tracker = tracker or OutcomeTracker()
    out: List[Outcome] = []
    for rec_id, override in overrides.items():
        if rec_id in already_measured:
            continue
        rec = recs_by_id.get(rec_id)
        if rec is None:
            continue
        if not is_outcome_measurable(override, rec, now_ts):
            continue
        outcome = measure_one(override, rec, process_data, tracker)
        if outcome is not None:
            out.append(outcome)
    return out


def outcome_summary_dict(outcome: Outcome, rec: Recommendation) -> Dict[str, Any]:
    """Lightweight serialization for the dashboard / decisions log."""
    return {
        "outcome_id":              outcome.id,
        "rec_id":                  rec.id,
        "rule_fired":              rec.rule_fired,
        "urgency":                 rec.urgency.value,
        "measured_at":             outcome.timestamp.isoformat(),
        "measurement_delay_min":   outcome.measurement_delay_minutes,
        "quality_score":           outcome.quality_score,
        "notes":                   outcome.notes,
        "metrics": [
            {
                "variable":          m.variable,
                "predicted_value":   m.predicted_value,
                "actual_value":      m.actual_value,
                "deviation_pct":     m.deviation_pct,
                "within_tolerance":  m.within_tolerance,
            }
            for m in outcome.metrics
        ],
    }
