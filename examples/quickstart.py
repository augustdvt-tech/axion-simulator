"""
Quickstart: build a custom simulation run programmatically.

This example shows how to configure a process from scratch. The same pattern
scales to arbitrary processes: subclass ProcessUnit for your units, subclass
Scenario for your disturbances, and the framework handles the rest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulator import (
    CSTR, DistillationColumn, Simulator, sequential_connections, Stream,
    SensorModel, DataLogger,
    NormalOperation, ThermalDrift, CompositeScenario,
)


# 1. Build process units — override any parameters you need
cstr_params = dict(CSTR.DEFAULT_PARAMETERS)
cstr_params["F"] = 2.5     # larger feed flow
cstr_params["V"] = 12.0    # larger reactor

column_params = dict(DistillationColumn.DEFAULT_PARAMETERS)
column_params["RR"] = 6.0  # extra reflux

cstr   = CSTR(name="reactor_1", parameters=cstr_params)
column = DistillationColumn(name="distillation_1", parameters=column_params)

# 2. Define feed stream
feed = Stream(
    name="fresh_feed",
    flow=cstr_params["F"],
    temperature=cstr_params["T_feed"],
    composition={"A": 1.0, "B": 0.0},
    pressure=2.0,
)

# 3. Build simulator with these units
sim = Simulator(
    units=[cstr, column],
    connections=sequential_connections(2),
    feed_stream=feed,
    dt=10.0,
    sample_period=60.0,
)

# 4. Attach a scenario, sensor model, and logger
scenario = CompositeScenario([
    NormalOperation(feed_flow_nominal=cstr_params["F"]),
    ThermalDrift(drift_duration_hours=48),
])

sensors = SensorModel(seed=42)
logger = DataLogger("data/custom_run.csv")

sim.attach_scenario(scenario)
sim.attach_sensors(sensors)
sim.attach_logger(logger)

# 5. Run the simulation
duration_hours = 24
print(f"Running custom simulation for {duration_hours} hours...")
sim.run(duration_seconds=duration_hours * 3600,
        start_timestamp=np.datetime64("2026-04-01T00:00:00"))
logger.close()
print("Done. Output written to data/custom_run.csv")
