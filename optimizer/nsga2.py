"""
Axion AI - Multi-Objective Optimizer (NSGA-II)
==============================================

Finds the Pareto front of operating points that balance multiple
conflicting process objectives (purity vs energy vs production etc).

Algorithm: NSGA-II
------------------
The Non-dominated Sorting Genetic Algorithm II is the de-facto standard
for multi-objective optimization in industrial process engineering. It
returns a *set* of solutions (the Pareto front) rather than a single
"best" — operators choose where on the trade-off curve they want to
operate.

Key concepts:
  - Domination: solution A dominates B if A is at least as good as B in
    every objective AND strictly better in at least one.
  - Non-dominated front: solutions that no other solution dominates.
  - Crowding distance: a measure of how isolated a solution is in
    objective space; we prefer well-spread fronts.

This implementation
-------------------
We hand-roll a minimal NSGA-II rather than depend on pymoo or platypus,
keeping the project's footprint small. The implementation is ~150 lines
of plain numpy and is deterministic given a seed.

Search space
------------
Each candidate solution is a vector of manipulated-variable values:
  [column.RR, cstr.F_cool, cstr.F_feed]

Bounds come from the SafetyLimits already used by the consensus controller.
The optimizer NEVER produces a candidate outside these bounds.

Disturbance variables (cstr.C_A, cstr.T_feed_C) are treated as fixed at
the values from the most recent process snapshot — we optimize FOR the
current operating context, not in the abstract.

Output: a Pareto front of `OptimizationResult` objects with the operating
point, predicted KPIs, and per-objective scores.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .surrogate import ProcessSurrogate
from .objectives import Objective, Direction


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class OperatingPoint:
    """A candidate setpoint vector + predicted KPIs."""
    inputs:    Dict[str, float]                      # setpoint values
    kpis:      Dict[str, float]                      # predicted process KPIs
    objectives: Dict[str, float] = field(default_factory=dict)   # per-objective evaluation
    rank:      Optional[int] = None                  # Pareto rank
    crowding:  Optional[float] = None                # crowding distance


# =============================================================================
# NSGA-II
# =============================================================================

class NSGA2Optimizer:
    """
    Multi-objective optimizer using NSGA-II.

    Typical usage:
        optimizer = NSGA2Optimizer(surrogate, objectives, bounds, fixed_inputs)
        front = optimizer.run(n_generations=40, population_size=80)
    """

    def __init__(
        self,
        surrogate: ProcessSurrogate,
        objectives: List[Objective],
        bounds: Dict[str, Tuple[float, float]],
        fixed_inputs: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.surrogate = surrogate
        self.objectives = objectives
        self.bounds = bounds
        self.fixed_inputs = fixed_inputs or {}
        self.rng = np.random.default_rng(seed)

        # Order of decision variables for the search
        self.var_names = [v for v in surrogate.inputs if v in bounds]
        # Sanity: every surrogate input must be either bounded or fixed
        missing = [v for v in surrogate.inputs
                   if v not in bounds and v not in self.fixed_inputs]
        if missing:
            raise ValueError(
                f"Surrogate inputs not covered by bounds or fixed_inputs: {missing}"
            )

    # ---- main loop ----

    def run(
        self,
        n_generations: int = 40,
        population_size: int = 80,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
    ) -> List[OperatingPoint]:
        """Run NSGA-II for the given number of generations. Returns the
        non-dominated front from the final population."""
        # Initial population (random within bounds)
        population = self._init_population(population_size)
        self._evaluate(population)

        for gen in range(n_generations):
            # Tournament selection of parents
            parents = self._tournament_select(population, n=population_size)
            # Crossover + mutation to produce offspring
            offspring = self._crossover_and_mutate(
                parents, crossover_prob, mutation_prob
            )
            self._evaluate(offspring)

            # Combine, rank, select top-N
            combined = population + offspring
            fronts = self._fast_non_dominated_sort(combined)
            population = self._select_next_generation(fronts, population_size)

        # Return the first non-dominated front
        final_fronts = self._fast_non_dominated_sort(population)
        front_0 = final_fronts[0]
        # Sort by primary objective for stable display
        front_0.sort(
            key=lambda p: p.objectives.get(self.objectives[0].name, 0.0)
        )
        return front_0

    # ---- helpers ----

    def _init_population(self, n: int) -> List[OperatingPoint]:
        pop: List[OperatingPoint] = []
        for _ in range(n):
            inputs = dict(self.fixed_inputs)
            for v in self.var_names:
                lo, hi = self.bounds[v]
                inputs[v] = float(self.rng.uniform(lo, hi))
            pop.append(OperatingPoint(inputs=inputs, kpis={}))
        return pop

    def _evaluate(self, points: List[OperatingPoint]) -> None:
        """Compute KPIs (via surrogate) and objective values for each point."""
        if not points:
            return
        # Batch surrogate prediction for efficiency
        X = pd.DataFrame([p.inputs for p in points])[self.surrogate.inputs]
        kpis_df = self.surrogate.predict(X)
        for p, (_, row) in zip(points, kpis_df.iterrows()):
            kpis = row.to_dict()
            # Inputs themselves are also relevant KPIs (production needs F_feed)
            for k in self.surrogate.inputs:
                kpis[k] = p.inputs[k]
            p.kpis = kpis
            p.objectives = {obj.name: obj.evaluate(kpis) for obj in self.objectives}

    def _signed_objectives(self, p: OperatingPoint) -> np.ndarray:
        """Return objective vector with sign convention 'lower is better'."""
        return np.array([
            obj.signed_value(p.kpis) for obj in self.objectives
        ])

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """A dominates B (lower-is-better convention)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _fast_non_dominated_sort(
        self, population: List[OperatingPoint]
    ) -> List[List[OperatingPoint]]:
        """Sort population into Pareto fronts (rank 0 = best)."""
        n = len(population)
        objs = [self._signed_objectives(p) for p in population]
        # For each solution: list of solutions it dominates, count of
        # solutions dominating it
        dominates: List[List[int]] = [[] for _ in range(n)]
        dominated_by: List[int] = [0] * n
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(objs[i], objs[j]):
                    dominates[i].append(j)
                elif self._dominates(objs[j], objs[i]):
                    dominated_by[i] += 1
            if dominated_by[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front: List[int] = []
            for i in fronts[k]:
                for j in dominates[i]:
                    dominated_by[j] -= 1
                    if dominated_by[j] == 0:
                        population[j].rank = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        fronts.pop()   # drop trailing empty

        # Convert index-fronts to point-fronts
        return [[population[i] for i in front] for front in fronts]

    def _crowding_distance(self, front: List[OperatingPoint]) -> None:
        """Assign crowding distance to each point in a front (in place)."""
        n = len(front)
        if n == 0:
            return
        for p in front:
            p.crowding = 0.0
        if n <= 2:
            for p in front:
                p.crowding = float("inf")
            return
        for obj_idx, obj in enumerate(self.objectives):
            values = np.array([obj.signed_value(p.kpis) for p in front])
            order = np.argsort(values)
            front[order[0]].crowding = float("inf")
            front[order[-1]].crowding = float("inf")
            v_min, v_max = values[order[0]], values[order[-1]]
            spread = v_max - v_min
            if spread <= 0:
                continue
            for i in range(1, n - 1):
                left = values[order[i - 1]]
                right = values[order[i + 1]]
                front[order[i]].crowding += (right - left) / spread

    def _select_next_generation(
        self, fronts: List[List[OperatingPoint]], size: int
    ) -> List[OperatingPoint]:
        """Select top-N for the next generation: prefer lower rank,
        then higher crowding distance."""
        out: List[OperatingPoint] = []
        for front in fronts:
            self._crowding_distance(front)
            if len(out) + len(front) <= size:
                out.extend(front)
            else:
                # Take the most crowded-distant ones from this front
                front_sorted = sorted(front, key=lambda p: -(p.crowding or 0.0))
                out.extend(front_sorted[: size - len(out)])
                break
        return out

    def _tournament_select(
        self, population: List[OperatingPoint], n: int, k: int = 2
    ) -> List[OperatingPoint]:
        """Binary tournament selection by (rank, -crowding)."""
        # Re-rank population if needed
        if any(p.rank is None for p in population):
            self._fast_non_dominated_sort(population)
            # Crowding distance per front
            from collections import defaultdict
            by_rank: Dict[int, List[OperatingPoint]] = defaultdict(list)
            for p in population:
                by_rank[p.rank].append(p)
            for front in by_rank.values():
                self._crowding_distance(front)

        chosen: List[OperatingPoint] = []
        for _ in range(n):
            candidates = self.rng.choice(len(population), size=k, replace=False)
            best = population[int(candidates[0])]
            for ci in candidates[1:]:
                c = population[int(ci)]
                if (c.rank, -(c.crowding or 0.0)) < (best.rank, -(best.crowding or 0.0)):
                    best = c
            chosen.append(best)
        return chosen

    def _crossover_and_mutate(
        self,
        parents: List[OperatingPoint],
        crossover_prob: float,
        mutation_prob: float,
    ) -> List[OperatingPoint]:
        """Simulated binary crossover + polynomial mutation, classic NSGA-II."""
        offspring: List[OperatingPoint] = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < len(parents) else parents[i]
            c1_inputs = dict(p1.inputs)
            c2_inputs = dict(p2.inputs)

            for v in self.var_names:
                lo, hi = self.bounds[v]
                # Crossover: blend within bounds
                if self.rng.random() < crossover_prob:
                    alpha = self.rng.uniform(0.3, 0.7)
                    new_v1 = alpha * p1.inputs[v] + (1 - alpha) * p2.inputs[v]
                    new_v2 = (1 - alpha) * p1.inputs[v] + alpha * p2.inputs[v]
                    c1_inputs[v] = float(np.clip(new_v1, lo, hi))
                    c2_inputs[v] = float(np.clip(new_v2, lo, hi))
                # Mutation: gaussian perturbation, scaled to range
                if self.rng.random() < mutation_prob:
                    scale = (hi - lo) * 0.05
                    c1_inputs[v] = float(np.clip(c1_inputs[v] + self.rng.normal(0, scale), lo, hi))
                if self.rng.random() < mutation_prob:
                    scale = (hi - lo) * 0.05
                    c2_inputs[v] = float(np.clip(c2_inputs[v] + self.rng.normal(0, scale), lo, hi))

            offspring.append(OperatingPoint(inputs=c1_inputs, kpis={}))
            offspring.append(OperatingPoint(inputs=c2_inputs, kpis={}))
        return offspring[: len(parents)]
