"""Tests for analytics/frozen.py: FrozenSensorDetector."""

import numpy as np
import pandas as pd
import pytest
from analytics import FrozenSensorDetector


@pytest.fixture
def frozen():
    return FrozenSensorDetector(
        tags=["cstr.T_R_C"],
        window_minutes=5,
        min_std=1e-4,
        cooldown_seconds=300,
    )


class TestFrozenSensorDetector:
    def test_fit_does_not_crash(self, frozen, df_synthetic):
        frozen.fit(df_synthetic)   # no-op, should not raise

    def test_no_alert_on_varying_signal(self, frozen, df_synthetic):
        frozen.fit(df_synthetic)
        alerts = frozen.run(df_synthetic)
        assert alerts == []

    def test_detects_constant_signal(self, frozen):
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.full(n, 78.0),
        })
        frozen.fit(df)
        alerts = frozen.run(df)
        assert len(alerts) >= 1

    def test_alert_detector_name(self, frozen):
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.full(n, 78.0),
        })
        frozen.fit(df)
        alerts = frozen.run(df)
        for a in alerts:
            assert a.detector == "FrozenSensor"

    def test_alert_tag_correct(self, frozen):
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.full(n, 78.0),
        })
        frozen.fit(df)
        alerts = frozen.run(df)
        for a in alerts:
            assert a.tag == "cstr.T_R_C"

    def test_missing_tag_does_not_crash(self):
        frozen = FrozenSensorDetector(tags=["nonexistent.tag"])
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="1min"),
            "cstr.T_R_C": np.full(10, 78.0),
        })
        alerts = frozen.run(df)
        assert alerts == []

    def test_cooldown_prevents_repeated_alerts(self, frozen):
        n = 60
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.full(n, 78.0),
        })
        frozen.fit(df)
        alerts = frozen.run(df)
        # With 300s cooldown and 1-min samples, should fire much less than once per row
        assert len(alerts) < n
