"""
Axion AI - Optimization Objectives
==================================

An Objective declares what we want the optimizer to maximize or minimize.
For multi-objective optimization, we keep each objective separate and let
NSGA-II find the Pareto front of non-dominated trade-offs.

Each objective has:
  - A direction (MAXIMIZE or MINIMIZE)
  - A method to compute its scalar value from a predicted KPI dict
  - A weight (optional, used for ε-constraint and operator preferences)
  - A normalization range (used for stability scoring and visualization)

The pilot process objectives are deliberately ones a process engineer would
recognize: maximize purity, minimize energy, maximize production, minimize
process variability. Adding a new objective for a different process is a
matter of subclassing Objective and adding it to the optimizer's list.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Objective(ABC):
    """Abstract base for any optimization objective."""

    def __init__(
        self,
        name: str,
        direction: Direction,
        weight: float = 1.0,
        nominal_range: Optional[Tuple[float, float]] = None,
        units: str = "",
    ):
        self.name = name
        self.direction = direction
        self.weight = weight
        self.nominal_range = nominal_range
        self.units = units

    @abstractmethod
    def evaluate(self, kpis: Dict[str, float]) -> float:
        """Return the raw value of this objective from the KPI dict."""
        ...

    def signed_value(self, kpis: Dict[str, float]) -> float:
        """
        Return the value with the sign convention that LOWER is BETTER.
        NSGA-II works in minimization; this normalization lets us mix
        maximize and minimize objectives uniformly.
        """
        v = self.evaluate(kpis)
        return -v if self.direction == Direction.MAXIMIZE else v


# =============================================================================
# Concrete objectives for the pilot process
# =============================================================================

class PurityObjective(Objective):
    """Maximize product purity. Penalizes harshly below spec."""

    def __init__(self, weight: float = 1.0, spec: float = 98.5):
        super().__init__(
            name="purity",
            direction=Direction.MAXIMIZE,
            weight=weight,
            nominal_range=(95.0, 99.8),
            units="%",
        )
        self.spec = spec

    def evaluate(self, kpis: Dict[str, float]) -> float:
        purity = kpis.get("column.purity_B", 0.0)
        # Heavy penalty below spec — multi-objective optimizer should see
        # "off-spec is much worse than just below high target"
        if purity < self.spec:
            penalty = (self.spec - purity) * 10.0
            return purity - penalty
        return purity


class EnergyObjective(Objective):
    """Minimize reboiler duty (the dominant energy cost in distillation)."""

    def __init__(self, weight: float = 1.0):
        super().__init__(
            name="energy",
            direction=Direction.MINIMIZE,
            weight=weight,
            nominal_range=(150.0, 350.0),
            units="kW",
        )

    def evaluate(self, kpis: Dict[str, float]) -> float:
        return kpis.get("column.Q_reb_kW", float("inf"))


class ProductionObjective(Objective):
    """Maximize production rate (proxy: F_feed × conversion of A → B)."""

    def __init__(self, weight: float = 0.7):
        super().__init__(
            name="production",
            direction=Direction.MAXIMIZE,
            weight=weight,
            nominal_range=(1.5, 2.5),
            units="kmol/h B",
        )

    def evaluate(self, kpis: Dict[str, float]) -> float:
        # Production rate of B = feed rate × conversion
        # F_feed in m³/h, with C_A_in ~ 8000 mol/m³
        F_feed = kpis.get("cstr.F_feed", 0.0)
        conversion = kpis.get("cstr.conversion", 0.0)
        return F_feed * conversion * 8.0


class StabilityObjective(Objective):
    """
    Penalize deviation from the reactor temperature setpoint. Lower is
    better — a stable reactor is more predictable and easier to operate.
    """

    def __init__(self, weight: float = 0.3, setpoint_T_R: float = 79.2):
        super().__init__(
            name="stability",
            direction=Direction.MINIMIZE,
            weight=weight,
            nominal_range=(0.0, 5.0),
            units="°C dev",
        )
        self.setpoint_T_R = setpoint_T_R

    def evaluate(self, kpis: Dict[str, float]) -> float:
        T_R = kpis.get("cstr.T_R_C", self.setpoint_T_R)
        return abs(T_R - self.setpoint_T_R)


# =============================================================================
# Default objective set for the pilot
# =============================================================================

PILOT_OBJECTIVES = [
    PurityObjective(weight=1.0),
    EnergyObjective(weight=1.0),
    ProductionObjective(weight=0.7),
    StabilityObjective(weight=0.3),
]
