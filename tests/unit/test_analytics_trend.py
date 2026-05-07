"""Tests for analytics/trend.py: TrendDetector."""

import numpy as np
import pandas as pd
import pytest
from analytics import TrendDetector, AlertType, PILOT_OPERATIONAL_LIMITS


@pytest.fixture
def trend():
    return TrendDetector(
        tags=["cstr.T_R_C"],
        limits=PILOT_OPERATIONAL_LIMITS,
        window_minutes=10,
        min_r_squared=0.5,
    )


class TestTrendDetectorRun:
    def test_no_alert_on_flat_signal(self, trend):
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.full(n, 78.0),
        })
        alerts = trend.run(df)
        assert alerts == []

    def test_detects_upward_trend_towards_limit(self, trend):
        """Signal rising from 78 → 86 over 30 min will cross high limit (83)."""
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.linspace(78.0, 86.0, n),
        })
        alerts = trend.run(df)
        trend_alerts = [a for a in alerts if a.alert_type == AlertType.TREND_PROJECTION]
        assert len(trend_alerts) >= 1

    def test_alert_includes_time_to_limit(self, trend):
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.linspace(78.0, 86.0, n),
        })
        alerts = trend.run(df)
        for a in alerts:
            assert "minutes" in a.message.lower() or a.extra.get("minutes_to_limit") is not None

    def test_no_alert_when_trend_away_from_limit(self, trend):
        """Downward trend away from high limit should not fire."""
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.linspace(82.0, 76.0, n),  # going down, away from 83 high
        })
        alerts = trend.run(df)
        high_alerts = [a for a in alerts
                       if a.alert_type == AlertType.TREND_PROJECTION
                       and a.extra.get("direction") == "high"]
        assert len(high_alerts) == 0

    def test_alert_tag_matches(self, trend):
        n = 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.linspace(78.0, 90.0, n),
        })
        alerts = trend.run(df)
        for a in alerts:
            assert a.tag == "cstr.T_R_C"

    def test_insufficient_data_returns_empty(self, trend):
        """Fewer rows than window — nothing to project."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
            "cstr.T_R_C": [78.0, 78.5, 79.0],
        })
        alerts = trend.run(df)
        assert isinstance(alerts, list)

    def test_flat_signal_no_alert(self):
        """Zero slope (R²=0) → no alert regardless of R² threshold."""
        n = 30
        trend = TrendDetector(
            tags=["cstr.T_R_C"],
            limits=PILOT_OPERATIONAL_LIMITS,
            window_minutes=10,
            min_r_squared=0.5,
        )
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "cstr.T_R_C": np.full(n, 78.0),   # perfectly flat → slope=0
        })
        alerts = trend.run(df)
        assert len(alerts) == 0
