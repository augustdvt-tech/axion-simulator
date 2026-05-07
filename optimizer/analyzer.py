"""
Axion AI - Operating Point Analyzer
===================================

Compares the current operating point against a pre-computed Pareto front
to detect optimization opportunities. Used as the data source for rule
R10_OptimizationOpportunity.

The analyzer answers:
  - Is the current operating point on (or near) the Pareto front?
  - If not, which Pareto-optimal point dominates it?
  - What's the operational improvement (purity, energy, production)?

Distance to the front
---------------------
We use a normalized weighted distance in objective space. Each objective
contributes proportionally to its `weight` and is normalized by its
`nominal_range`. A point with distance > threshold is considered far from
the front and triggers a recommendation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .nsga2 import OperatingPoint
from .objectives import Objective, Direction


@dataclass
class OptimizationGap:
    """Difference between current operation and the closest dominating Pareto point."""
    current_kpis:    Dict[str, float]
    best_pareto:     OperatingPoint
    distance:        float                 # normalized
    improvements:    Dict[str, float]      # by objective, signed (positive = improvement)
    proposed_inputs: Dict[str, float]      # the setpoint values to move toward


class OperatingPointAnalyzer:
    """
    Analyzes the current operating point against a Pareto front of
    candidate solutions, producing an OptimizationGap if the current
    point is dominated.
    """

    def __init__(
        self,
        objectives: List[Objective],
        improvement_threshold: float = 0.05,   # min relative improvement to flag
    ):
        self.objectives = objectives
        self.improvement_threshold = improvement_threshold

    def analyze(
        self,
        current_kpis: Dict[str, float],
        pareto_front: List[OperatingPoint],
    ) -> Optional[OptimizationGap]:
        """
        Find the Pareto point that dominates the current operation by the
        largest normalized margin. Returns None if no Pareto point dominates
        the current point (i.e. we're already on the front).
        """
        if not pareto_front:
            return None

        # Current point's objective vector (lower is better convention)
        current_signed = np.array([
            obj.signed_value(current_kpis) for obj in self.objectives
        ])

        # Find dominating Pareto points
        best_dom: Optional[OperatingPoint] = None
        best_score = 0.0
        for p in pareto_front:
            p_signed = np.array([
                obj.signed_value(p.kpis) for obj in self.objectives
            ])
            # p dominates current if better-or-equal on all and strictly
            # better on at least one
            if not np.all(p_signed <= current_signed) or not np.any(p_signed < current_signed):
                continue
            # Weighted normalized improvement score
            score = 0.0
            for i, obj in enumerate(self.objectives):
                rng = obj.nominal_range
                if rng is None:
                    span = max(abs(current_signed[i]), 1e-6)
                else:
                    span = max(abs(rng[1] - rng[0]), 1e-6)
                gain = (current_signed[i] - p_signed[i]) / span
                score += obj.weight * max(gain, 0.0)
            if score > best_score:
                best_score = score
                best_dom = p

        if best_dom is None:
            return None
        if best_score < self.improvement_threshold:
            return None

        improvements = {}
        for obj in self.objectives:
            current_val = obj.evaluate(current_kpis)
            best_val    = obj.evaluate(best_dom.kpis)
            if obj.direction == Direction.MAXIMIZE:
                improvements[obj.name] = best_val - current_val
            else:
                improvements[obj.name] = current_val - best_val

        return OptimizationGap(
            current_kpis=current_kpis,
            best_pareto=best_dom,
            distance=best_score,
            improvements=improvements,
            proposed_inputs=dict(best_dom.inputs),
        )
