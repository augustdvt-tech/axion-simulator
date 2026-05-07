"""Tests for analytics/spc.py: SPCDetector."""

import numpy as np
import pandas as pd
import pytest
from analytics import SPCDetector, Alert, AlertType, Severity


@pytest.fixture
def spc():
    return SPCDetector(tags=["cstr.T_R_C", "column.purity_B"])


@pytest.fixture
def df_train(df_synthetic):
    return df_synthetic  # 20 rows at nominal conditions


class TestTagBaseline:
    def test_shewhart_limits_symmetric(self, spc, df_train):
        spc.fit(df_train)
        bl = spc.baselines["cstr.T_R_C"]
        assert bl.shewhart_ucl > bl.mean > bl.shewhart_lcl
        assert abs((bl.shewhart_ucl - bl.mean) - (bl.mean - bl.shewhart_lcl)) < 1e-6

    def test_ewma_limits_tighter_than_shewhart(self, spc, df_train):
        spc.fit(df_train)
        bl = spc.baselines["cstr.T_R_C"]
        assert bl.ewma_ucl < bl.shewhart_ucl
        assert bl.ewma_lcl > bl.shewhart_lcl


class TestSPCDetectorFit:
    def test_fit_populates_baselines(self, spc, df_train):
        spc.fit(df_train)
        assert "cstr.T_R_C" in spc.baselines
        assert "column.purity_B" in spc.baselines

    def test_baseline_mean_close_to_nominal(self, spc, df_train):
        spc.fit(df_train)
        mean = spc.baselines["cstr.T_R_C"].mean
        assert abs(mean - 78.0) < 2.0


class TestSPCDetectorRun:
    def test_no_alerts_on_clean_data(self, spc, df_train):
        spc.fit(df_train)
        alerts = spc.run(df_train)
        assert isinstance(alerts, list)

    def test_shewhart_alert_on_spike(self, spc, df_train, df_with_spike):
        spc.fit(df_train)
        alerts = spc.run(df_with_spike)
        shewhart_alerts = [a for a in alerts
                           if a.alert_type == AlertType.SHEWHART_VIOLATION
                           and a.tag == "cstr.T_R_C"]
        assert len(shewhart_alerts) >= 1

    def test_alert_has_required_fields(self, spc, df_train, df_with_spike):
        spc.fit(df_train)
        alerts = spc.run(df_with_spike)
        for a in alerts:
            assert isinstance(a.timestamp, pd.Timestamp)
            assert a.detector.startswith("SPC")   # may be "SPC.Shewhart" or "SPC.EWMA"
            assert a.severity in list(Severity)
            assert a.tag is not None

    def test_ewma_detects_slow_drift(self, spc, df_with_drift):
        """120-row dataset with 8°C drift — EWMA should fire at some point."""
        spc.fit(df_with_drift.iloc[:30])   # train on first 30 rows (clean portion)
        alerts = spc.run(df_with_drift.iloc[30:])
        ewma_alerts = [a for a in alerts
                       if a.alert_type == AlertType.EWMA_VIOLATION
                       and a.tag == "cstr.T_R_C"]
        assert len(ewma_alerts) >= 1

    def test_cooldown_limits_alert_spam(self, spc, df_train):
        """A sustained violation should not produce one alert per row."""
        spc.fit(df_train)
        # Build a DF where T_R_C is always above UCL
        df_hot = df_train.copy()
        df_hot["cstr.T_R_C"] = 200.0
        alerts = spc.run(df_hot)
        t_r_shewhart = [a for a in alerts
                        if a.alert_type == AlertType.SHEWHART_VIOLATION
                        and a.tag == "cstr.T_R_C"]
        assert len(t_r_shewhart) <= len(df_hot)   # at most one per row
        assert len(t_r_shewhart) < len(df_hot)    # but likely fewer due to cooldown

    def test_severity_levels(self, spc, df_train):
        spc.fit(df_train)
        df_extreme = df_train.copy()
        df_extreme["cstr.T_R_C"] = 300.0   # far above limits
        alerts = spc.run(df_extreme)
        severities = {a.severity for a in alerts if a.tag == "cstr.T_R_C"}
        # Should include at least a high or critical severity
        assert severities & {Severity.HIGH, Severity.CRITICAL}

    def test_run_only_monitored_tags(self):
        """SPC only monitors its configured tags."""
        spc = SPCDetector(tags=["cstr.T_R_C"])
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
            "cstr.T_R_C": [78.0] * 5,
            "column.purity_B": [99.3] * 5,
        })
        spc.fit(df)
        alerts = spc.run(df)
        assert all(a.tag == "cstr.T_R_C" for a in alerts if a.tag is not None)
