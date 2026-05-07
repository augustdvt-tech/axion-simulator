"""Tests for optimizer/objectives.py: Objective implementations."""

import pytest
from optimizer import (
    PurityObjective, EnergyObjective, ProductionObjective, StabilityObjective,
    Direction, PILOT_OBJECTIVES,
)


NOMINAL_KPIS = {
    "column.purity_B": 98.97,
    "column.Q_reb_kW": 235.0,
    "cstr.conversion": 0.84,
    "cstr.T_R_C": 78.86,
    "cstr.F_feed": 2.0,
}


class TestPurityObjective:
    def test_above_spec_returns_purity(self):
        obj = PurityObjective(spec=98.5)
        score = obj.evaluate({"column.purity_B": 99.0})
        assert score == pytest.approx(99.0)

    def test_below_spec_penalized(self):
        obj = PurityObjective(spec=98.5)
        score = obj.evaluate({"column.purity_B": 97.0})
        assert score < 97.0  # penalty applied

    def test_further_below_lower_score(self):
        obj = PurityObjective(spec=98.5)
        s1 = obj.evaluate({"column.purity_B": 98.0})
        s2 = obj.evaluate({"column.purity_B": 95.0})
        assert s2 < s1  # further below spec → lower (worse) score for MAXIMIZE

    def test_direction_is_maximize(self):
        obj = PurityObjective()
        assert obj.direction == Direction.MAXIMIZE


class TestEnergyObjective:
    def test_lower_energy_lower_score(self):
        obj = EnergyObjective()
        s_low  = obj.evaluate({"column.Q_reb_kW": 200.0})
        s_high = obj.evaluate({"column.Q_reb_kW": 300.0})
        assert s_low < s_high

    def test_direction_is_minimize(self):
        obj = EnergyObjective()
        assert obj.direction == Direction.MINIMIZE

    def test_positive_score(self):
        obj = EnergyObjective()
        assert obj.evaluate(NOMINAL_KPIS) >= 0.0


class TestProductionObjective:
    def test_higher_conversion_higher_score(self):
        obj = ProductionObjective()
        s_high = obj.evaluate({"cstr.conversion": 0.9, "cstr.F_feed": 2.0})
        s_low  = obj.evaluate({"cstr.conversion": 0.7, "cstr.F_feed": 2.0})
        assert s_high > s_low

    def test_direction_is_maximize(self):
        obj = ProductionObjective()
        assert obj.direction == Direction.MAXIMIZE


class TestStabilityObjective:
    def test_at_setpoint_low_score(self):
        obj = StabilityObjective(setpoint_T_R=79.2)
        score = obj.evaluate({"cstr.T_R_C": 79.2})
        assert score == pytest.approx(0.0)

    def test_deviation_increases_score(self):
        obj = StabilityObjective(setpoint_T_R=79.2)
        s1 = obj.evaluate({"cstr.T_R_C": 80.0})
        s2 = obj.evaluate({"cstr.T_R_C": 85.0})
        assert s2 > s1


class TestPilotObjectives:
    def test_all_four_present(self):
        assert len(PILOT_OBJECTIVES) == 4

    def test_all_have_evaluate_method(self):
        for obj in PILOT_OBJECTIVES:
            assert hasattr(obj, "evaluate")
            assert callable(obj.evaluate)

    def test_all_evaluate_nominal_without_error(self):
        for obj in PILOT_OBJECTIVES:
            score = obj.evaluate(NOMINAL_KPIS)
            assert isinstance(score, float)
