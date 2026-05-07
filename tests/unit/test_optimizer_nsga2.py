"""Tests for optimizer/nsga2.py: NSGA2Optimizer (slow — runs genetic algorithm)."""

import pytest
from optimizer import (
    ProcessSurrogate, NSGA2Optimizer, OperatingPoint,
    PurityObjective, EnergyObjective, ProductionObjective, StabilityObjective,
)


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pareto_front():
    surrogate = ProcessSurrogate()
    bounds = {
        "column.RR":   (3.0, 7.5),
        "cstr.F_cool": (0.10, 0.55),
        "cstr.F_feed": (1.7, 2.3),
    }
    fixed = {"cstr.C_A": 157.0, "cstr.T_feed_C": 70.0}
    optimizer = NSGA2Optimizer(
        surrogate=surrogate,
        objectives=[PurityObjective(spec=98.5), EnergyObjective(),
                    ProductionObjective(), StabilityObjective()],
        bounds=bounds,
        fixed_inputs=fixed,
        seed=42,
    )
    return optimizer.run(n_generations=10, population_size=20)


class TestNSGA2Optimizer:
    def test_returns_list_of_operating_points(self, pareto_front):
        assert isinstance(pareto_front, list)
        assert len(pareto_front) > 0
        assert all(isinstance(p, OperatingPoint) for p in pareto_front)

    def test_operating_points_have_inputs(self, pareto_front):
        for pt in pareto_front:
            assert "column.RR" in pt.inputs
            assert "cstr.F_cool" in pt.inputs

    def test_operating_points_have_kpis(self, pareto_front):
        for pt in pareto_front:
            assert "column.purity_B" in pt.kpis
            assert "column.Q_reb_kW" in pt.kpis

    def test_inputs_within_bounds(self, pareto_front):
        for pt in pareto_front:
            assert 3.0 <= pt.inputs["column.RR"] <= 7.5
            assert 0.10 <= pt.inputs["cstr.F_cool"] <= 0.55
            assert 1.7 <= pt.inputs["cstr.F_feed"] <= 2.3

    def test_deterministic_with_same_seed(self):
        surrogate = ProcessSurrogate()
        bounds = {"column.RR": (4.0, 7.0), "cstr.F_cool": (0.20, 0.45),
                  "cstr.F_feed": (1.8, 2.2)}
        fixed = {"cstr.C_A": 157.0, "cstr.T_feed_C": 70.0}

        def run():
            opt = NSGA2Optimizer(
                surrogate=surrogate,
                objectives=[PurityObjective(), EnergyObjective()],
                bounds=bounds, fixed_inputs=fixed, seed=7,
            )
            front = opt.run(n_generations=5, population_size=10)
            return sorted(pt.inputs["column.RR"] for pt in front)

        assert run() == run()
