"""
Axion AI - Recommendation System
================================

Consumes event sessions from the Analytical Engine and produces structured
operator-facing recommendations via a rule-based diagnostic library.

Primary entry point:
    RecommendationEngine.generate(sessions, process_data) -> List[Recommendation]
"""

from .models import (
    Recommendation, Action, ActionType, ExpectedImpact, Urgency,
    recommendations_to_dataframe,
)
from .rules_base import DiagnosticRule, RuleContext
from .rules_pilot import (
    PILOT_RULES,
    R01_ThermalDrift, R02_FeedComposition, R03_ControllerOscillation,
    R04_ColumnEfficiencyLoss, R05_ExcessReflux, R06_PurityDeviation,
    R07_SensorFault, R08_ProductTransition, R09_SoftSensorDivergence,
    R10_PredictedExcursion,
)
from .engine import RecommendationEngine

__all__ = [
    "Recommendation", "Action", "ActionType", "ExpectedImpact", "Urgency",
    "recommendations_to_dataframe",
    "DiagnosticRule", "RuleContext",
    "PILOT_RULES",
    "R01_ThermalDrift", "R02_FeedComposition", "R03_ControllerOscillation",
    "R04_ColumnEfficiencyLoss", "R05_ExcessReflux", "R06_PurityDeviation",
    "R07_SensorFault", "R08_ProductTransition", "R09_SoftSensorDivergence",
    "R10_PredictedExcursion",
    "RecommendationEngine",
]
