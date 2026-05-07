"""
Axion AI - Simulation Runner
============================

CLI entry point that runs any configured scenario end-to-end and produces
a CSV file suitable for ingestion by the Axion AI analytics engine.

Usage:
    python run_simulation.py --scenario normal --duration-hours 24 --output data/normal.csv
    python run_simulation.py --scenario thermal_drift --duration-hours 72 --output data/drift.csv
    python run_simulation.py --scenario all        # runs all scenarios
"""

from __future__ import annotations
import argparse
import time
import numpy as np
from pathlib import Path

from simulator import (
    CSTR, DistillationColumn, Simulator, sequential_connections, Stream,
    SensorModel, DataLogger,
    NormalOperation, ThermalDrift, FeedPerturbation, ReactorInstability,
    QualityDegradation, EnergyWaste, ProductGradeChange, SensorFailureScenario,
    CompositeScenario,
)


# =============================================================================
# Builder: assemble the pilot process (CSTR + Column)
# =============================================================================

def build_pilot_process() -> Simulator:
    """Assemble the pilot process: CSTR followed by binary distillation."""
    cstr = CSTR(name="cstr", parameters=CSTR.DEFAULT_PARAMETERS)
    column = DistillationColumn(name="column",
                                parameters=DistillationColumn.DEFAULT_PARAMETERS)

    # Warm-up: run each unit to steady state before starting the scenario
    warm_up(cstr, column, hours=10)

    feed = Stream(
        name="fresh_feed",
        flow=CSTR.DEFAULT_PARAMETERS["F"],
        temperature=CSTR.DEFAULT_PARAMETERS["T_feed"],
        composition={"A": 1.0, "B": 0.0},
        pressure=2.0,
    )

    sim = Simulator(
        units=[cstr, column],
        connections=sequential_connections(2),
        feed_stream=feed,
        dt=10.0,             # 10 s integration step
        sample_period=60.0,  # 1 sample per minute
    )
    return sim


def warm_up(cstr: CSTR, column: DistillationColumn, hours: float = 10.0) -> None:
    """Integrate both units forward long enough to reach steady state."""
    dt = 10.0
    n_steps = int(hours * 3600 / dt)

    feed = Stream(
        name="fresh_feed",
        flow=cstr.parameters["F"],
        temperature=cstr.parameters["T_feed"],
        composition={"A": 1.0, "B": 0.0},
        pressure=2.0,
    )
    cstr_out = Stream(name="cstr_out")

    for step in range(n_steps):
        t = step * dt
        cstr.step_rk4(t, dt, feed)
        cstr.compute_outlet(cstr.state, feed, cstr_out)
        column.step_rk4(t, dt, cstr_out)


# =============================================================================
# Scenario factory
# =============================================================================

def get_scenario(name: str, sensor_model: SensorModel):
    """Build a scenario instance (possibly composite with NormalOperation)."""
    normal = NormalOperation(
        feed_flow_nominal=CSTR.DEFAULT_PARAMETERS["F"], seed=42
    )

    if name == "normal":
        return normal
    if name == "thermal_drift":
        return CompositeScenario([normal, ThermalDrift()])
    if name == "feed_perturbation":
        return CompositeScenario([normal, FeedPerturbation()])
    if name == "reactor_instability":
        return CompositeScenario([normal, ReactorInstability()])
    if name == "quality_degradation":
        return CompositeScenario([normal, QualityDegradation()])
    if name == "energy_waste":
        return CompositeScenario([normal, EnergyWaste()])
    if name == "product_grade_change":
        return CompositeScenario([normal, ProductGradeChange()])
    if name == "sensor_failure":
        return CompositeScenario([
            normal,
            SensorFailureScenario(sensor_model, "T_R_C",
                                  failure_type="frozen",
                                  start_hours=8, duration_hours=4),
        ])
    raise ValueError(f"Unknown scenario: {name}")


# =============================================================================
# Runner
# =============================================================================

def run_scenario(scenario_name: str, duration_hours: float, output_path: str,
                 start_timestamp: str = "2026-04-01T00:00:00",
                 quiet: bool = False) -> None:
    t_start = time.time()

    sim = build_pilot_process()
    sensors = SensorModel(seed=123)
    scenario = get_scenario(scenario_name, sensors)
    logger = DataLogger(output_path)

    sim.attach_scenario(scenario)
    sim.attach_sensors(sensors)
    sim.attach_logger(logger)

    t0 = np.datetime64(start_timestamp)
    duration_s = duration_hours * 3600.0

    sim.run(duration_seconds=duration_s, start_timestamp=t0)
    logger.close()

    elapsed = time.time() - t_start
    n_samples = int(duration_s / sim.sample_period)
    if not quiet:
        print(f"  ✓ scenario '{scenario_name}'  |  "
              f"{duration_hours:.0f} h simulated  |  "
              f"{n_samples} samples  |  "
              f"runtime {elapsed:.1f} s  |  "
              f"→ {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Axion AI process simulator")
    parser.add_argument("--scenario", default="normal",
                        help="Scenario name, or 'all' to run all scenarios")
    parser.add_argument("--duration-hours", type=float, default=24.0,
                        help="Simulation duration in hours (default: 24)")
    parser.add_argument("--output", default="data/simulation.csv",
                        help="Output CSV path")
    parser.add_argument("--start-timestamp", default="2026-04-01T00:00:00",
                        help="Timestamp for the first sample (ISO format)")
    args = parser.parse_args()

    if args.scenario == "all":
        scenarios = [
            ("normal", 24),
            ("thermal_drift", 72),
            ("feed_perturbation", 24),
            ("reactor_instability", 24),
            ("quality_degradation", 96),
            ("energy_waste", 24),
            ("product_grade_change", 24),
            ("sensor_failure", 24),
        ]
        print("Running all scenarios:")
        for name, hours in scenarios:
            run_scenario(name, hours, f"data/{name}.csv",
                         args.start_timestamp)
    else:
        run_scenario(args.scenario, args.duration_hours, args.output,
                     args.start_timestamp)


if __name__ == "__main__":
    main()
