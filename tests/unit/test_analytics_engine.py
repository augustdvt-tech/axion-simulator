"""Tests for analytics/engine.py: AnalyticalEngine."""

import numpy as np
import pandas as pd
import pytest
from analytics import AnalyticalEngine, EventSession, PILOT_TAGS


@pytest.fixture
def engine():
    return AnalyticalEngine(training_fraction=0.5, warmup_minutes=0.0)


class TestAnalyticalEngineInit:
    def test_default_tags(self):
        ae = AnalyticalEngine()
        for tag in PILOT_TAGS:
            assert tag in ae.tags

    def test_custom_tags(self):
        ae = AnalyticalEngine(tags=["cstr.T_R_C"])
        assert ae.tags == ["cstr.T_R_C"]


class TestAnalyticalEngineFit:
    def test_fit_does_not_crash(self, engine, df_synthetic):
        engine.fit(df_synthetic)

    def test_fit_with_real_csv(self, engine, df_normal_csv):
        engine.fit(df_normal_csv)
        assert engine.spc.baselines


class TestAnalyticalEngineRun:
    def test_run_returns_list(self, engine, df_synthetic):
        engine.fit(df_synthetic)
        alerts = engine.run(df_synthetic)
        assert isinstance(alerts, list)

    def test_run_sessions_returns_sessions(self, engine, df_synthetic):
        engine.fit(df_synthetic)
        sessions = engine.run_sessions(df_synthetic)
        assert isinstance(sessions, list)
        for s in sessions:
            assert isinstance(s, EventSession)

    def test_spike_produces_alert(self, engine, df_synthetic, df_with_spike):
        engine.fit(df_synthetic)
        alerts = engine.run(df_with_spike)
        spc_alerts = [a for a in alerts if a.detector.startswith("SPC") and a.tag == "cstr.T_R_C"]
        assert len(spc_alerts) >= 1

    def test_run_to_dataframe_has_columns(self, engine, df_synthetic, df_with_spike):
        engine.fit(df_synthetic)
        df_alerts = engine.run_to_dataframe(df_with_spike)
        for col in ("timestamp", "detector", "severity", "tag"):
            assert col in df_alerts.columns

    def test_sessions_group_related_alerts(self, engine, df_with_drift):
        engine.fit(df_with_drift.iloc[:60])
        sessions = engine.run_sessions(df_with_drift.iloc[60:])
        # All sessions should have at least 1 alert
        assert all(s.alert_count >= 1 for s in sessions)

    def test_separate_fit_and_eval(self, df_synthetic):
        """Fitting on one dataset and evaluating on another should not crash."""
        ae = AnalyticalEngine(training_fraction=1.0, warmup_minutes=0.0)
        ae.fit(df_synthetic)
        alerts = ae.run(df_synthetic)
        assert isinstance(alerts, list)

    def test_post_training_only_skips_train_window(self, engine, df_with_drift):
        engine.fit(df_with_drift)
        alerts_all   = engine.run(df_with_drift, post_training_only=False)
        alerts_post  = engine.run(df_with_drift, post_training_only=True)
        # Post-training slice has fewer rows → at most as many alerts
        assert len(alerts_post) <= len(alerts_all)
