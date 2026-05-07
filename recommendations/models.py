"""
Axion AI - Recommendation Data Model
====================================

The output of the Recommendation System. A Recommendation is what the
operator sees: not raw statistics, but a structured suggestion they can
evaluate, approve, modify or reject.

Design principle: every recommendation must answer five operator questions:

    1. WHAT is happening?           -> diagnosis
    2. WHY is it happening?         -> probable_cause
    3. WHAT should I do?            -> action
    4. WHAT will happen if I do?    -> expected_impact
    5. HOW CONFIDENT is the system? -> confidence + urgency

Additionally, each recommendation carries a full audit trail:
    - Which EventSession(s) triggered it
    - Which rule or pattern fired
    - Timestamp, priority score, unique ID for decision logging

This structure is what the Human-Machine Consensus module (future Task 5)
will track across approve/modify/reject decisions.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd
import uuid


class Urgency(str, Enum):
    LOW = "low"           # > 2 hours to impact
    MEDIUM = "medium"     # 30 min - 2 hours
    HIGH = "high"         # 10 - 30 min
    CRITICAL = "critical" # < 10 min, immediate action needed


class ActionType(str, Enum):
    """What kind of operator action is being proposed."""
    ADJUST_SETPOINT = "adjust_setpoint"       # change a manipulated variable
    INVESTIGATE = "investigate"               # look into something manually
    VERIFY_INSTRUMENT = "verify_instrument"   # check sensor calibration
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    WAIT_AND_MONITOR = "wait_and_monitor"     # no action, keep watching
    ESCALATE = "escalate"                     # call supervisor / engineer


@dataclass
class Action:
    """
    A specific action the operator can take. Structured to be both
    human-readable and machine-executable (when in semi-autonomous mode).
    """
    type: ActionType
    description: str                     # plain text for operators
    target_variable: Optional[str] = None    # e.g. "column.RR"
    current_value: Optional[float] = None
    proposed_value: Optional[float] = None
    adjustment: Optional[float] = None       # delta from current
    units: Optional[str] = None              # e.g. "dimensionless", "m3/h"

    @property
    def is_automated(self) -> bool:
        """True if this action could in principle be executed automatically."""
        return (
            self.type == ActionType.ADJUST_SETPOINT
            and self.target_variable is not None
            and self.proposed_value is not None
        )


@dataclass
class ExpectedImpact:
    """
    The predicted effect of executing the action. The Recommendation
    Engine populates this from process knowledge (rules) or from a model.
    """
    variable: str                  # e.g. "column.purity_B"
    current_value: Optional[float]
    predicted_value: Optional[float]
    time_to_effect_minutes: Optional[float] = None
    description: str = ""


@dataclass
class Recommendation:
    """
    The unit of output from the Recommendation Engine. One recommendation
    per diagnosed situation.

    Lifecycle:
        created (by engine) -> reviewed -> (accepted | modified | rejected) -> executed -> outcome
    """
    # Identity
    id: str
    timestamp: pd.Timestamp
    priority_score: float              # 0..100, for sorting in UI

    # The 5 operator questions
    diagnosis: str                     # WHAT
    probable_cause: str                # WHY
    action: Action                     # WHAT TO DO
    expected_impact: List[ExpectedImpact]   # WHAT WILL HAPPEN
    confidence: float                  # 0..1, HOW SURE
    urgency: Urgency                   # HOW URGENT

    # Audit trail
    triggering_sessions: List[str] = field(default_factory=list)  # session identifiers
    rule_fired: str = ""               # name of the rule that produced this
    affected_variables: List[str] = field(default_factory=list)

    # Status (updated by the consensus module)
    status: str = "pending"            # pending | accepted | modified | rejected | executed

    # Extra metadata — free-form dict for rule-specific info
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return f"REC-{uuid.uuid4().hex[:8].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["urgency"] = self.urgency.value
        d["action"]["type"] = self.action.type.value
        d["timestamp"] = (
            self.timestamp.isoformat()
            if hasattr(self.timestamp, "isoformat")
            else str(self.timestamp)
        )
        return d

    def format_summary(self) -> str:
        """One-line summary for logs and terminal output."""
        return (
            f"[{self.urgency.value.upper():<8s}] "
            f"{self.diagnosis[:70]} "
            f"(priority {self.priority_score:.0f}, conf {self.confidence:.0%})"
        )

    def format_detail(self) -> str:
        """Multi-line detailed view for the recommendation card UI."""
        lines = [
            f"╭─ RECOMMENDATION {self.id}",
            f"│  Urgency: {self.urgency.value.upper()}  |  "
            f"Priority: {self.priority_score:.0f}  |  "
            f"Confidence: {self.confidence:.0%}",
            f"│  Time: {self.timestamp}",
            f"│",
            f"│  DIAGNOSIS      → {self.diagnosis}",
            f"│  PROBABLE CAUSE → {self.probable_cause}",
            f"│",
            f"│  RECOMMENDED ACTION:",
            f"│     {self.action.description}",
        ]
        if self.action.target_variable and self.action.proposed_value is not None:
            adj_str = (
                f" (Δ = {self.action.adjustment:+.3f})"
                if self.action.adjustment is not None
                else ""
            )
            lines.append(
                f"│     {self.action.target_variable}: "
                f"{self.action.current_value:.3f} → {self.action.proposed_value:.3f}"
                f" {self.action.units or ''}{adj_str}"
            )
        if self.expected_impact:
            lines.append(f"│")
            lines.append(f"│  EXPECTED IMPACT:")
            for imp in self.expected_impact:
                if imp.current_value is not None and imp.predicted_value is not None:
                    arrow = f"{imp.current_value:.3f} → {imp.predicted_value:.3f}"
                else:
                    arrow = imp.description
                eta = (
                    f" in ~{imp.time_to_effect_minutes:.0f} min"
                    if imp.time_to_effect_minutes else ""
                )
                lines.append(f"│     {imp.variable}: {arrow}{eta}")
        lines.append(f"│")
        lines.append(f"│  Rule: {self.rule_fired}")
        lines.append(f"╰─")
        return "\n".join(lines)


def recommendations_to_dataframe(recs: List[Recommendation]) -> pd.DataFrame:
    """Flatten a list of Recommendations to a pandas DataFrame for analysis."""
    if not recs:
        return pd.DataFrame(columns=[
            "id", "timestamp", "urgency", "priority_score", "confidence",
            "diagnosis", "probable_cause", "action_type", "action_description",
            "target_variable", "current_value", "proposed_value",
            "rule_fired", "status",
        ])
    rows = []
    for r in recs:
        rows.append({
            "id":                r.id,
            "timestamp":         r.timestamp,
            "urgency":           r.urgency.value,
            "priority_score":    r.priority_score,
            "confidence":        r.confidence,
            "diagnosis":         r.diagnosis,
            "probable_cause":    r.probable_cause,
            "action_type":       r.action.type.value,
            "action_description": r.action.description,
            "target_variable":   r.action.target_variable,
            "current_value":     r.action.current_value,
            "proposed_value":    r.action.proposed_value,
            "rule_fired":        r.rule_fired,
            "status":            r.status,
        })
    return pd.DataFrame(rows).sort_values(
        ["timestamp", "priority_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
