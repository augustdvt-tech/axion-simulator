"""
Axion AI - Consensus & Decision Layer
=====================================

Closes the human-machine loop:
- Captures every operator decision with full justification and audit trail.
- Measures the actual outcome of executed actions vs. predicted impact.
- Tracks per-rule performance to feed continuous learning.
- Enforces three operating modes (advisor / semi-autonomous / autonomous)
  with safety gates and rollback policies.

Primary entry point: ConsensusController.process(recommendations, process_data)
"""

from .models import (
    Decision, DecisionStatus, Execution, ExecutionStatus,
    Outcome, OutcomeMetric,
    decisions_to_dataframe, outcomes_to_dataframe,
)
from .operator import (
    OperatorPolicy, RealisticOperator, ConservativeOperator, AggressiveOperator,
)
from .outcome import (
    OutcomeTracker, OutcomeTrackerConfig,
    PerformanceTracker, RulePerformance,
)
from .controller import (
    ConsensusController, OperatingMode, SafetyLimits, AutonomyRules,
    DecisionLog, PILOT_SAFETY_LIMITS,
)

__all__ = [
    # models
    "Decision", "DecisionStatus", "Execution", "ExecutionStatus",
    "Outcome", "OutcomeMetric",
    "decisions_to_dataframe", "outcomes_to_dataframe",
    # operator
    "OperatorPolicy", "RealisticOperator", "ConservativeOperator", "AggressiveOperator",
    # outcome
    "OutcomeTracker", "OutcomeTrackerConfig",
    "PerformanceTracker", "RulePerformance",
    # controller
    "ConsensusController", "OperatingMode", "SafetyLimits", "AutonomyRules",
    "DecisionLog", "PILOT_SAFETY_LIMITS",
]
