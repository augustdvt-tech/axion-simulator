"""
Axion AI - Simulator Core
=========================

Base abstractions for the process simulator. Designed to be extended to any
industrial process by subclassing ProcessUnit and connecting units via Streams.

Design principles:
- Physics-driven: each unit defines its own ODE derivatives
- Modular: units are independent, connected only through material streams
- Scalable: adding a new unit = subclassing ProcessUnit
- Reproducible: deterministic given seeds and initial conditions
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np


# =============================================================================
# Stream: material flow between units
# =============================================================================

@dataclass
class Stream:
    """
    Represents a material stream connecting two process units.

    For the MVP we track the fields relevant to a binary A+B system. For other
    industries, extend with additional fields (pressure, enthalpy, multi-component
    compositions, etc.) — the base Simulator doesn't care about the contents.
    """
    name: str
    flow: float = 0.0           # Volumetric flow [m3/h]
    temperature: float = 298.15  # Temperature [K]
    composition: Dict[str, float] = field(default_factory=dict)  # mole fractions
    pressure: float = 1.013      # Pressure [bar]

    def copy_from(self, other: "Stream") -> None:
        """Copy state from another stream (used to propagate between units)."""
        self.flow = other.flow
        self.temperature = other.temperature
        self.composition = dict(other.composition)
        self.pressure = other.pressure


# =============================================================================
# ProcessUnit: base class for any process unit
# =============================================================================

class ProcessUnit(ABC):
    """
    Abstract base class for any process unit (reactor, column, tank, HX, etc.).

    Subclasses must implement:
    - state_variables: list of state variable names
    - derivatives(t, x, inlet): returns dx/dt given current state and inlet stream
    - compute_outlet(x, inlet): fills the outlet stream from current state
    - initial_state(): returns steady-state x0
    - measured_variables(x, inlet): returns dict of variables that sensors see
    """

    def __init__(self, name: str, parameters: Dict):
        self.name = name
        self.parameters = dict(parameters)   # mutable: scenarios can modify
        self._initial_parameters = dict(parameters)  # for reset
        self.state = np.array(self.initial_state(), dtype=float)

    # --- subclass interface ---
    @property
    @abstractmethod
    def state_variables(self) -> List[str]:
        ...

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        ...

    @abstractmethod
    def derivatives(self, t: float, x: np.ndarray, inlet: Stream) -> np.ndarray:
        ...

    @abstractmethod
    def compute_outlet(self, x: np.ndarray, inlet: Stream, outlet: Stream) -> None:
        ...

    @abstractmethod
    def measured_variables(self, x: np.ndarray, inlet: Stream) -> Dict[str, float]:
        ...

    # --- framework interface ---
    def reset_parameters(self) -> None:
        """Restore original parameters (used between scenario runs)."""
        self.parameters = dict(self._initial_parameters)
        self.state = np.array(self.initial_state(), dtype=float)

    def step_rk4(self, t: float, dt: float, inlet: Stream) -> None:
        """Advance state one step using 4th order Runge-Kutta."""
        x = self.state
        k1 = self.derivatives(t,          x,             inlet)
        k2 = self.derivatives(t + dt / 2, x + dt * k1 / 2, inlet)
        k3 = self.derivatives(t + dt / 2, x + dt * k2 / 2, inlet)
        k4 = self.derivatives(t + dt,     x + dt * k3,     inlet)
        self.state = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


# =============================================================================
# Simulator: orchestrates the process units
# =============================================================================

class Simulator:
    """
    Time-marching orchestrator for a connected set of process units.

    The simulator integrates each unit's ODEs with a fixed RK4 step, propagates
    material streams between units in order, applies scenario perturbations, and
    logs measured variables + applied sensor noise to a DataLogger.
    """

    def __init__(
        self,
        units: List[ProcessUnit],
        connections: List[tuple],   # (source_unit_idx, dest_unit_idx)
        feed_stream: Stream,        # external feed into unit 0
        dt: float = 10.0,           # integration step [s]
        sample_period: float = 60.0,  # data output period [s]
    ):
        self.units = units
        self.connections = connections
        self.feed = feed_stream
        self.dt = dt
        self.sample_period = sample_period

        # Create streams between units
        self.streams: Dict[int, Stream] = {}
        for i, u in enumerate(units):
            self.streams[i] = Stream(name=f"{u.name}_out")

        # Scenario and sensor hooks
        self.scenario = None        # optional Scenario instance
        self.sensor_model = None    # optional SensorModel instance
        self.logger = None          # optional DataLogger instance

    def attach_scenario(self, scenario) -> None:
        self.scenario = scenario

    def attach_sensors(self, sensor_model) -> None:
        self.sensor_model = sensor_model

    def attach_logger(self, logger) -> None:
        self.logger = logger

    def _unit_inlet(self, unit_idx: int) -> Stream:
        """Determine the inlet stream for a given unit."""
        # Find incoming connection, if any
        for src, dst in self.connections:
            if dst == unit_idx:
                return self.streams[src]
        # No incoming connection → external feed
        return self.feed

    def run(self, duration_seconds: float, start_timestamp=None) -> None:
        """
        Run the simulation for the given duration.

        start_timestamp: optional pandas.Timestamp / datetime to use as t0 in logs.
        """
        n_steps = int(duration_seconds / self.dt)
        sample_every = int(self.sample_period / self.dt)

        for step in range(n_steps):
            t = step * self.dt

            # 1) Apply scenario perturbations (modify feed, parameters, etc.)
            if self.scenario is not None:
                self.scenario.apply(t=t, feed=self.feed, units=self.units)

            # 2) Integrate each unit one step, then propagate outlet
            for i, unit in enumerate(self.units):
                inlet = self._unit_inlet(i)
                unit.step_rk4(t, self.dt, inlet)
                unit.compute_outlet(unit.state, inlet, self.streams[i])

            # 3) Sample and log
            if step % sample_every == 0 and self.logger is not None:
                row = {"time_s": t}
                if start_timestamp is not None:
                    row["timestamp"] = start_timestamp + np.timedelta64(int(t), "s")

                # Gather measured variables from each unit
                for i, unit in enumerate(self.units):
                    inlet = self._unit_inlet(i)
                    measurements = unit.measured_variables(unit.state, inlet)
                    for name, value in measurements.items():
                        tag = f"{unit.name}.{name}"
                        # Apply sensor model (noise / bias / failures)
                        if self.sensor_model is not None:
                            value = self.sensor_model.apply(tag, value, t)
                        row[tag] = value

                self.logger.log(row)


# =============================================================================
# Utility: connect two units sequentially
# =============================================================================

def sequential_connections(n_units: int) -> List[tuple]:
    """Helper: build a linear connection list for n sequential units."""
    return [(i, i + 1) for i in range(n_units - 1)]
