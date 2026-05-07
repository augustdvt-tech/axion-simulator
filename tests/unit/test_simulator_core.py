"""Tests for simulator/core.py: Stream, ProcessUnit, Simulator, sequential_connections."""

import numpy as np
import pytest
from simulator import Stream, Simulator, sequential_connections, CSTR, DistillationColumn


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class TestStream:
    def test_default_values(self):
        s = Stream(name="feed")
        assert s.flow == 0.0
        assert s.temperature == 298.15
        assert s.pressure == 1.013
        assert s.composition == {}

    def test_copy_from(self):
        src = Stream("src", flow=2.0, temperature=343.0,
                     composition={"A": 1.0}, pressure=2.5)
        dst = Stream("dst")
        dst.copy_from(src)
        assert dst.flow == 2.0
        assert dst.temperature == 343.0
        assert dst.composition == {"A": 1.0}
        assert dst.pressure == 2.5

    def test_copy_from_does_not_share_composition(self):
        src = Stream("src", composition={"A": 0.9, "B": 0.1})
        dst = Stream("dst")
        dst.copy_from(src)
        dst.composition["A"] = 0.5
        assert src.composition["A"] == 0.9   # src is unchanged


# ---------------------------------------------------------------------------
# sequential_connections
# ---------------------------------------------------------------------------

class TestSequentialConnections:
    def test_two_units(self):
        assert sequential_connections(2) == [(0, 1)]

    def test_three_units(self):
        assert sequential_connections(3) == [(0, 1), (1, 2)]

    def test_one_unit(self):
        assert sequential_connections(1) == []


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class TestSimulator:
    @pytest.fixture
    def feed(self):
        return Stream(
            name="feed", flow=2.0, temperature=343.15,
            composition={"A": 1.0, "B": 0.0}, pressure=2.0,
        )

    @pytest.fixture
    def simulator(self, feed, tmp_path):
        cstr = CSTR(name="cstr", parameters=CSTR.DEFAULT_PARAMETERS)
        col  = DistillationColumn(name="column",
                                  parameters=DistillationColumn.DEFAULT_PARAMETERS)
        sim  = Simulator(
            units=[cstr, col],
            connections=sequential_connections(2),
            feed_stream=feed,
            dt=10.0,
            sample_period=60.0,
        )
        return sim

    def test_attach_scenario(self, simulator):
        from simulator import NormalOperation
        scenario = NormalOperation(seed=0)
        simulator.attach_scenario(scenario)
        assert simulator.scenario is scenario

    def test_attach_sensors(self, simulator):
        from simulator import SensorModel
        sensors = SensorModel(seed=1)
        simulator.attach_sensors(sensors)
        assert simulator.sensor_model is sensors

    def test_run_produces_samples(self, simulator, tmp_path):
        from simulator import DataLogger, SensorModel
        out = tmp_path / "out.csv"
        simulator.attach_sensors(SensorModel(seed=42))
        simulator.attach_logger(DataLogger(str(out)))
        simulator.run(duration_seconds=120.0,
                      start_timestamp=np.datetime64("2026-01-01T00:00:00"))
        simulator.logger.close()
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 2    # 120s / 60s sample_period
        assert "cstr.T_R_C" in df.columns

    def test_run_deterministic(self, feed, tmp_path):
        """Same seed → same output on two independent runs."""
        from simulator import DataLogger, SensorModel

        def run_once(path):
            cstr = CSTR(name="cstr", parameters=CSTR.DEFAULT_PARAMETERS)
            col  = DistillationColumn(name="column",
                                      parameters=DistillationColumn.DEFAULT_PARAMETERS)
            sim  = Simulator(
                units=[cstr, col],
                connections=sequential_connections(2),
                feed_stream=Stream(
                    name="feed", flow=2.0, temperature=343.15,
                    composition={"A": 1.0}, pressure=2.0,
                ),
                dt=10.0, sample_period=60.0,
            )
            sim.attach_sensors(SensorModel(seed=99))
            sim.attach_logger(DataLogger(str(path)))
            sim.run(60.0, start_timestamp=np.datetime64("2026-01-01T00:00:00"))
            sim.logger.close()

        import pandas as pd
        p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
        run_once(p1)
        run_once(p2)
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        assert (df1["cstr.T_R_C"].values == df2["cstr.T_R_C"].values).all()
