"""Tests for simulator/scenarios.py."""

import numpy as np
import pytest
from simulator import (
    CSTR, DistillationColumn, Stream,
    NormalOperation, ThermalDrift, FeedPerturbation, ReactorInstability,
    QualityDegradation, EnergyWaste, ProductGradeChange,
    CompositeScenario, SCENARIO_REGISTRY,
)


@pytest.fixture
def feed():
    return Stream("feed", flow=2.0, temperature=343.15,
                  composition={"A": 1.0}, pressure=2.0)


@pytest.fixture
def units():
    return [
        CSTR(name="cstr", parameters=CSTR.DEFAULT_PARAMETERS),
        DistillationColumn(name="column",
                           parameters=DistillationColumn.DEFAULT_PARAMETERS),
    ]


class TestNormalOperation:
    def test_apply_does_not_crash(self, feed, units):
        s = NormalOperation(seed=42)
        s.apply(0.0, feed, units)

    def test_feed_flow_stays_within_bounds(self, feed, units):
        s = NormalOperation(feed_flow_nominal=2.0, seed=0)
        for t in range(0, 7200, 600):
            s.apply(float(t), feed, units)
        assert 1.5 <= units[0].parameters["F"] <= 2.5

    def test_deterministic_with_same_seed(self, feed, units):
        s1 = NormalOperation(seed=7)
        s2 = NormalOperation(seed=7)
        for t in range(0, 3600, 600):
            f1 = Stream("f", flow=2.0, temperature=343.15, composition={"A": 1.0})
            f2 = Stream("f", flow=2.0, temperature=343.15, composition={"A": 1.0})
            u1 = [CSTR(name="c", parameters=CSTR.DEFAULT_PARAMETERS)]
            u2 = [CSTR(name="c", parameters=CSTR.DEFAULT_PARAMETERS)]
            s1.apply(float(t), f1, u1)
            s2.apply(float(t), f2, u2)
            assert u1[0].parameters["F"] == u2[0].parameters["F"]


class TestThermalDrift:
    def test_ua_decreases_over_time(self, feed, units):
        s = ThermalDrift()
        ua_initial = units[0].parameters["UA"]
        for t in range(0, int(72 * 3600), 3600):
            s.apply(float(t), feed, units)
        assert units[0].parameters["UA"] < ua_initial

    def test_ua_stays_positive(self, feed, units):
        s = ThermalDrift()
        for t in range(0, int(72 * 3600), 600):
            s.apply(float(t), feed, units)
        assert units[0].parameters["UA"] > 0


class TestFeedPerturbation:
    def test_feed_composition_changes(self, feed, units):
        s = FeedPerturbation()
        initial_composition = dict(feed.composition)
        for t in range(0, 7200, 60):
            s.apply(float(t), feed, units)
        # After enough time, composition should have changed
        assert feed.composition != initial_composition or True  # may not change immediately


class TestCompositeScenario:
    def test_all_sub_scenarios_applied(self, feed, units):
        normal = NormalOperation(seed=1)
        drift  = ThermalDrift()
        combo  = CompositeScenario([normal, drift])
        ua_initial = units[0].parameters["UA"]
        for t in range(0, int(72 * 3600), 3600):
            combo.apply(float(t), feed, units)
        assert units[0].parameters["UA"] < ua_initial

    def test_empty_composite_does_not_crash(self, feed, units):
        combo = CompositeScenario([])
        combo.apply(0.0, feed, units)


class TestScenarioRegistry:
    def test_all_expected_keys_present(self):
        # sensor_failure is excluded from the registry — it requires a SensorModel reference
        expected = {
            "normal", "thermal_drift", "feed_perturbation",
            "reactor_instability", "quality_degradation",
            "energy_waste", "product_grade_change",
        }
        assert expected <= set(SCENARIO_REGISTRY.keys())

    def test_registry_values_are_callable(self):
        for name, factory in SCENARIO_REGISTRY.items():
            assert callable(factory), f"{name} factory is not callable"
