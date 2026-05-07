"""Tests for simulator/instrumentation.py: SensorModel and DataLogger."""

import numpy as np
import pytest
from simulator import SensorModel, DataLogger


class TestSensorModel:
    def test_apply_adds_noise(self):
        sm = SensorModel(seed=42)
        values = [sm.apply("cstr.T_R_C", 78.0, t) for t in range(20)]
        assert not all(v == 78.0 for v in values)

    def test_noise_is_small(self):
        sm = SensorModel(seed=0)
        deviations = [abs(sm.apply("cstr.T_R_C", 78.0, t) - 78.0) for t in range(50)]
        assert max(deviations) < 5.0    # < 5 degC noise is reasonable

    def test_deterministic_with_same_seed(self):
        sm1 = SensorModel(seed=99)
        sm2 = SensorModel(seed=99)
        v1 = [sm1.apply("cstr.T_R_C", 78.0, float(t)) for t in range(10)]
        v2 = [sm2.apply("cstr.T_R_C", 78.0, float(t)) for t in range(10)]
        assert v1 == v2

    def test_frozen_failure_returns_constant(self):
        sm = SensorModel(seed=1)
        sm.apply("cstr.T_R_C", 78.0, 0.0)   # prime the frozen value
        sm.inject_failure("T_R_C", "frozen")
        first = sm.apply("cstr.T_R_C", 80.0, 60.0)
        second = sm.apply("cstr.T_R_C", 85.0, 120.0)
        assert first == second

    def test_drift_failure_increases_over_time(self):
        sm = SensorModel(seed=2)
        sm.inject_failure("T_R_C", "drift")
        v0 = sm.apply("cstr.T_R_C", 78.0, 0.0)
        v1 = sm.apply("cstr.T_R_C", 78.0, 3600.0)
        assert v1 > v0

    def test_clear_failure_restores_normal_noise(self):
        sm = SensorModel(seed=3)
        sm.inject_failure("T_R_C", "frozen")
        sm.apply("cstr.T_R_C", 78.0, 0.0)
        sm.clear_failure("T_R_C")
        values = [sm.apply("cstr.T_R_C", 78.0, t) for t in range(10)]
        assert len(set(values)) > 1    # no longer frozen

    def test_unknown_tag_uses_default_noise(self):
        sm = SensorModel(seed=5)
        v = sm.apply("some.unknown_tag", 100.0, 0.0)
        assert isinstance(v, float)

    def test_bias_failure_offsets_value(self):
        sm = SensorModel(seed=6)
        sm.inject_failure("T_R_C", "bias")
        readings = [sm.apply("cstr.T_R_C", 78.0, t) for t in range(5)]
        # After bias is set, all readings should be offset
        assert all(r != 78.0 for r in readings[1:])


class TestDataLogger:
    def test_log_and_close_creates_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        logger = DataLogger(str(path))
        logger.log({"time_s": 0.0, "cstr.T_R_C": 78.0, "column.purity_B": 99.3})
        logger.log({"time_s": 60.0, "cstr.T_R_C": 78.1, "column.purity_B": 99.2})
        logger.close()
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) == 2
        assert "cstr.T_R_C" in df.columns

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "subdir" / "nested" / "out.csv"
        logger = DataLogger(str(path))
        logger.log({"time_s": 0.0, "x": 1.0})
        logger.close()
        assert path.exists()

    def test_consistent_columns_across_rows(self, tmp_path):
        path = tmp_path / "out.csv"
        logger = DataLogger(str(path))
        logger.log({"time_s": 0.0, "a": 1.0, "b": 2.0})
        logger.log({"time_s": 60.0, "a": 1.1, "b": 2.1})
        logger.close()
        import pandas as pd
        df = pd.read_csv(path)
        assert list(df.columns) == ["time_s", "a", "b"]
