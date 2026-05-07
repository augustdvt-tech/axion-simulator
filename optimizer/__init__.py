"""
Axion AI - Multi-Objective Optimizer
====================================

Finds operating points that balance conflicting process objectives
(purity vs energy vs production etc) using a fast surrogate model
of the process plus NSGA-II.

Components:
  - ProcessSurrogate:    fast ML proxy of the process (KPI prediction)
  - Objective + concrete: declarative objectives (PurityObjective, etc)
  - NSGA2Optimizer:      multi-objective genetic algorithm → Pareto front
"""

from .surrogate import (
    ProcessSurrogate, SurrogateMetrics,
    SURROGATE_INPUTS, SURROGATE_OUTPUTS,
)
from .objectives import (
    Objective, Direction,
    PurityObjective, EnergyObjective, ProductionObjective, StabilityObjective,
    PILOT_OBJECTIVES,
)
from .nsga2 import NSGA2Optimizer, OperatingPoint
from .analyzer import OperatingPointAnalyzer, OptimizationGap

__all__ = [
    "ProcessSurrogate", "SurrogateMetrics",
    "SURROGATE_INPUTS", "SURROGATE_OUTPUTS",
    "Objective", "Direction",
    "PurityObjective", "EnergyObjective", "ProductionObjective", "StabilityObjective",
    "PILOT_OBJECTIVES",
    "NSGA2Optimizer", "OperatingPoint",
    "OperatingPointAnalyzer", "OptimizationGap",
]
