"""
Axion AI - Decision Log Models
==============================

Data model for the audit trail of every interaction between Axion AI and the
operator. This is the foundation of the consensus system and the source of
data for continuous learning.

The lifecycle of a recommendation from creation to outcome measurement:

    Recommendation (created by RecommendationEngine, status=pending)
            ↓
    Decision (operator reviews and decides)
        - status: accepted | modified | rejected | deferred
        - if modified: actual_action carries the operator's amended values
        - justification: free text from the operator
            ↓
    Execution (only if accepted/modified, action is then carried out)
        - executed_at, executor (human or system), final_action_values
            ↓
    Outcome (measured N minutes after execution)
        - actual values of impacted variables vs. predicted values
        - deviation_pct: how off was the prediction?
        - rated_quality: was the action ultimately a good one?

Design choices:

- Decisions are STORED PERMANENTLY. Even rejected recommendations are kept
  because they tell us something about which rules the operator distrusts.

- The Outcome is a SEPARATE record, not a field on the Decision. This lets
  outcome measurement happen asynchronously (e.g. 30-60 min after execution)
  and avoids blocking the consensus flow.

- All structures serialize cleanly to JSON / database rows so this can plug
  into any persistence backend in the future. For the MVP we keep them in
  memory and serialize to CSV.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd
import uuid

from recommendations import Recommendation, Action


class DecisionStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"          # operator approved as-is
    MODIFIED = "modified"          # operator changed the values, then approved
    REJECTED = "rejected"          # operator dismissed the recommendation
    DEFERRED = "deferred"          # operator postponed (review later)
    AUTO_EXECUTED = "auto_executed"  # autonomous mode handled it directly
    EXPIRED = "expired"            # not acted on within validity window


class ExecutionStatus(str, Enum):
    NOT_REQUIRED = "not_required"  # for INVESTIGATE/WAIT actions, no setpoint write
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"   # rollback policy triggered (see consensus)


# =============================================================================
# Decision: the operator's response to a recommendation
# =============================================================================

@dataclass
class Decision:
    """
    Records the operator's response to a Recommendation.

    For ACCEPTED: actual_action mirrors the original recommendation's action.
    For MODIFIED: actual_action carries the operator's amended values.
    For REJECTED / DEFERRED: actual_action is None.
    """
    id: str
    recommendation_id: str
    timestamp: pd.Timestamp                # when the decision was made
    operator_id: str                       # who decided (or "system" for autonomous)
    status: DecisionStatus
    actual_action: Optional[Action] = None
    justification: str = ""                # required for MODIFIED / REJECTED

    # Linkage to recommendation snapshot for forensic analysis
    rec_snapshot: Optional[Dict[str, Any]] = None  # full Recommendation.to_dict()

    @staticmethod
    def new_id() -> str:
        return f"DEC-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id":                self.id,
            "recommendation_id": self.recommendation_id,
            "timestamp":         self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "operator_id":       self.operator_id,
            "status":            self.status.value,
            "justification":     self.justification,
        }
        if self.actual_action is not None:
            d["actual_action"] = {
                "type":             self.actual_action.type.value,
                "description":      self.actual_action.description,
                "target_variable":  self.actual_action.target_variable,
                "current_value":    self.actual_action.current_value,
                "proposed_value":   self.actual_action.proposed_value,
                "adjustment":       self.actual_action.adjustment,
                "units":            self.actual_action.units,
            }
        else:
            d["actual_action"] = None
        return d


# =============================================================================
# Execution: the record of carrying out an approved action
# =============================================================================

@dataclass
class Execution:
    """
    Records the actual execution of an approved action. For action types
    that don't write to the process (INVESTIGATE, WAIT_AND_MONITOR), the
    execution record still exists with status=NOT_REQUIRED for traceability.
    """
    id: str
    decision_id: str
    timestamp: pd.Timestamp
    status: ExecutionStatus
    executor: str = "operator"             # "operator" | "system" | "rollback"
    error_message: str = ""

    @staticmethod
    def new_id() -> str:
        return f"EXE-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":            self.id,
            "decision_id":   self.decision_id,
            "timestamp":     self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "status":        self.status.value,
            "executor":      self.executor,
            "error_message": self.error_message,
        }


# =============================================================================
# Outcome: the post-action measurement
# =============================================================================

@dataclass
class OutcomeMetric:
    """The measured outcome on a single impacted variable."""
    variable: str
    predicted_value: Optional[float]
    actual_value: Optional[float]
    measurement_delay_minutes: float
    deviation_abs: Optional[float] = None     # |actual - predicted|
    deviation_pct: Optional[float] = None     # |dev| / |predicted - baseline|
    within_tolerance: Optional[bool] = None   # was prediction within tolerance?

    def __post_init__(self):
        if self.predicted_value is None or self.actual_value is None:
            return
        self.deviation_abs = abs(self.actual_value - self.predicted_value)
        denom = max(abs(self.predicted_value), 1e-6)
        self.deviation_pct = self.deviation_abs / denom


@dataclass
class Outcome:
    """
    Records the observed outcome of an executed action, measured a fixed
    number of minutes after execution.

    quality_score is a single 0..1 number summarizing 'how well did the
    action work?' — averaged across all metrics, weighted by their
    operational importance.
    """
    id: str
    execution_id: str
    decision_id: str
    recommendation_id: str
    timestamp: pd.Timestamp                # when the outcome was measured
    measurement_delay_minutes: float        # how long after execution
    metrics: List[OutcomeMetric] = field(default_factory=list)
    quality_score: float = 0.0             # 0..1 overall rating
    notes: str = ""

    @staticmethod
    def new_id() -> str:
        return f"OUT-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                        self.id,
            "execution_id":              self.execution_id,
            "decision_id":               self.decision_id,
            "recommendation_id":         self.recommendation_id,
            "timestamp":                 self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "measurement_delay_minutes": self.measurement_delay_minutes,
            "quality_score":             self.quality_score,
            "notes":                     self.notes,
            "metrics": [
                {
                    "variable":              m.variable,
                    "predicted_value":       m.predicted_value,
                    "actual_value":          m.actual_value,
                    "measurement_delay_minutes": m.measurement_delay_minutes,
                    "deviation_abs":         m.deviation_abs,
                    "deviation_pct":         m.deviation_pct,
                    "within_tolerance":      m.within_tolerance,
                } for m in self.metrics
            ],
        }


# =============================================================================
# DataFrame helpers
# =============================================================================

def decisions_to_dataframe(decisions: List[Decision]) -> pd.DataFrame:
    if not decisions:
        return pd.DataFrame(columns=[
            "id", "recommendation_id", "timestamp", "operator_id",
            "status", "justification",
            "action_type", "target_variable", "proposed_value",
        ])
    rows = []
    for d in decisions:
        row = {
            "id":                d.id,
            "recommendation_id": d.recommendation_id,
            "timestamp":         d.timestamp,
            "operator_id":       d.operator_id,
            "status":            d.status.value,
            "justification":     d.justification,
            "action_type":       d.actual_action.type.value if d.actual_action else None,
            "target_variable":   d.actual_action.target_variable if d.actual_action else None,
            "proposed_value":    d.actual_action.proposed_value if d.actual_action else None,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def outcomes_to_dataframe(outcomes: List[Outcome]) -> pd.DataFrame:
    if not outcomes:
        return pd.DataFrame(columns=[
            "id", "decision_id", "recommendation_id", "timestamp",
            "quality_score", "n_metrics", "mean_deviation_pct",
        ])
    rows = []
    for o in outcomes:
        valid_metrics = [m for m in o.metrics if m.deviation_pct is not None]
        mean_dev = (
            sum(m.deviation_pct for m in valid_metrics) / len(valid_metrics)
            if valid_metrics else None
        )
        rows.append({
            "id":                  o.id,
            "decision_id":         o.decision_id,
            "recommendation_id":   o.recommendation_id,
            "timestamp":           o.timestamp,
            "quality_score":       o.quality_score,
            "n_metrics":           len(o.metrics),
            "mean_deviation_pct":  mean_dev,
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
