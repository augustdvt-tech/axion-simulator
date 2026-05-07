"""
Axion AI - Operating Scenarios
==============================

Library of operating scenarios. A Scenario perturbs the process (feed, jacket
fouling, sensor failure, operator setpoint, etc.) as a function of time to
generate data that challenges the analytical engine.

Scenarios are composable: you can combine a base 'Normal' scenario with a
'SensorFailure' overlay to generate realistic data for training/testing.

Adding a new scenario = subclass Scenario and implement apply().
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .core import Stream, ProcessUnit


# =============================================================================
# Base class
# =============================================================================

class Scenario(ABC):
    """A scenario perturbs the simulation at each timestep."""

    name: str = "base"

    @abstractmethod
    def apply(self, t: float, feed: Stream, units: List[ProcessUnit]) -> None:
        """Called every integration step. Modify feed or unit.parameters as needed."""
        ...


class CompositeScenario(Scenario):
    """Combines multiple scenarios (applied in order)."""

    name = "composite"

    def __init__(self, scenarios: List[Scenario]):
        self.scenarios = scenarios

    def apply(self, t, feed, units):
        for s in self.scenarios:
            s.apply(t, feed, units)


# =============================================================================
# Scenario 1: Normal operation
# =============================================================================

class NormalOperation(Scenario):
    """Baseline: nominal operation with small stochastic load variations."""

    name = "normal_operation"

    def __init__(self, feed_flow_nominal: float = 2.0, seed: int = 42):
        self.feed_flow_nominal = feed_flow_nominal
        self.rng = np.random.default_rng(seed)
        self._last_update = -1e9

    def apply(self, t, feed, units):
        # Update load every 10 minutes with slow random walk
        if t - self._last_update > 600:
            delta = self.rng.normal(0, 0.03)
            new_flow = np.clip(self.feed_flow_nominal * (1 + delta), 1.8, 2.2)
            units[0].parameters["F"] = new_flow
            self._last_update = t


# =============================================================================
# Scenario 2: Thermal drift (jacket fouling)
# =============================================================================

class ThermalDrift(Scenario):
    """
    Gradual fouling of the cooling jacket: UA degrades linearly over time.
    Simulates heat exchanger fouling, a very common slow-developing issue.
    """

    name = "thermal_drift"

    def __init__(self, ua_initial: float = 4.5e4, ua_final: float = 3.0e4,
                 drift_duration_hours: float = 72.0):
        self.ua0 = ua_initial
        self.ua_f = ua_final
        self.drift_secs = drift_duration_hours * 3600.0

    def apply(self, t, feed, units):
        progress = min(t / self.drift_secs, 1.0)
        ua = self.ua0 + (self.ua_f - self.ua0) * progress
        units[0].parameters["UA"] = ua


# =============================================================================
# Scenario 3: Feed composition perturbation
# =============================================================================

class FeedPerturbation(Scenario):
    """
    Step change of +15% in feed concentration of A at a specified time.
    Tests response to unmeasured disturbances.
    """

    name = "feed_perturbation"

    def __init__(self, step_time_hours: float = 12.0, magnitude_pct: float = 15.0):
        self.step_time = step_time_hours * 3600.0
        self.magnitude = magnitude_pct / 100.0
        self._applied = False
        self._original = None

    def apply(self, t, feed, units):
        if self._original is None:
            self._original = units[0].parameters["C_A0"]
        if t >= self.step_time and not self._applied:
            units[0].parameters["C_A0"] = self._original * (1 + self.magnitude)
            self._applied = True


# =============================================================================
# Scenario 4: Reactor instability (oscillatory behavior)
# =============================================================================

class ReactorInstability(Scenario):
    """
    Induces oscillations in reactor temperature by sinusoidally modulating
    coolant flow, mimicking a poorly tuned PID controller.
    """

    name = "reactor_instability"

    def __init__(self, start_hours: float = 6.0, period_minutes: float = 20.0,
                 amplitude_fraction: float = 0.25):
        self.start = start_hours * 3600.0
        self.period = period_minutes * 60.0
        self.amp = amplitude_fraction
        self._nominal = None

    def apply(self, t, feed, units):
        if self._nominal is None:
            self._nominal = units[0].parameters["F_c"]
        if t >= self.start:
            phase = 2 * np.pi * (t - self.start) / self.period
            units[0].parameters["F_c"] = self._nominal * (1 + self.amp * np.sin(phase))


# =============================================================================
# Scenario 5: Quality degradation
# =============================================================================

class QualityDegradation(Scenario):
    """
    Gradual degradation of column performance: relative volatility decreases
    (simulates tray fouling or thermosyphon issues).
    """

    name = "quality_degradation"

    def __init__(self, alpha_initial: float = 3.2, alpha_final: float = 2.3,
                 drift_duration_hours: float = 96.0):
        self.a0 = alpha_initial
        self.af = alpha_final
        self.duration = drift_duration_hours * 3600.0

    def apply(self, t, feed, units):
        # units[1] is the column
        if len(units) > 1:
            progress = min(t / self.duration, 1.0)
            alpha = self.a0 + (self.af - self.a0) * progress
            units[1].parameters["alpha"] = alpha


# =============================================================================
# Scenario 6: Energy waste (excess reflux)
# =============================================================================

class EnergyWaste(Scenario):
    """
    Operator runs the column with excess reflux ratio. Quality is fine but
    energy consumption is unnecessarily high — opportunity for optimization.
    """

    name = "energy_waste"

    def __init__(self, start_hours: float = 4.0, excess_rr: float = 1.5):
        self.start = start_hours * 3600.0
        self.excess = excess_rr
        self._applied = False
        self._original = None

    def apply(self, t, feed, units):
        if len(units) > 1 and self._original is None:
            self._original = units[1].parameters["RR"]
        if t >= self.start and not self._applied and len(units) > 1:
            units[1].parameters["RR"] = self._original + self.excess
            self._applied = True


# =============================================================================
# Scenario 7: Sensor failure
# =============================================================================

class SensorFailureScenario(Scenario):
    """
    Scenario wrapper that signals the sensor model to fail a specific tag.
    The actual failure behavior is implemented in the SensorModel — this
    scenario just schedules it.
    """

    name = "sensor_failure"

    def __init__(self, sensor_model, tag: str, failure_type: str = "frozen",
                 start_hours: float = 8.0, duration_hours: float = 4.0):
        self.sensor_model = sensor_model
        self.tag = tag
        self.failure_type = failure_type
        self.start = start_hours * 3600.0
        self.end = self.start + duration_hours * 3600.0
        self._applied = False
        self._ended = False

    def apply(self, t, feed, units):
        if t >= self.start and not self._applied:
            self.sensor_model.inject_failure(self.tag, self.failure_type)
            self._applied = True
        if t >= self.end and not self._ended:
            self.sensor_model.clear_failure(self.tag)
            self._ended = True


# =============================================================================
# Scenario 8: Product grade change
# =============================================================================

class ProductGradeChange(Scenario):
    """
    Operator changes product specification: new setpoint for reflux ratio
    to achieve a different target purity. Tests transition handling.
    """

    name = "product_grade_change"

    def __init__(self, change_time_hours: float = 10.0,
                 new_rr: float = 4.0):
        self.change_time = change_time_hours * 3600.0
        self.new_rr = new_rr
        self._applied = False

    def apply(self, t, feed, units):
        if t >= self.change_time and not self._applied and len(units) > 1:
            units[1].parameters["RR"] = self.new_rr
            self._applied = True


# =============================================================================
# Registry
# =============================================================================

SCENARIO_REGISTRY = {
    "normal":           NormalOperation,
    "thermal_drift":    ThermalDrift,
    "feed_perturbation": FeedPerturbation,
    "reactor_instability": ReactorInstability,
    "quality_degradation": QualityDegradation,
    "energy_waste":     EnergyWaste,
    "product_grade_change": ProductGradeChange,
    # sensor_failure requires sensor_model reference, not in default registry
}
