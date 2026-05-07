"""Tests for recommendations/engine.py: RecommendationEngine."""

import pandas as pd
import numpy as np
import pytest
from analytics import (
    Alert, AlertType, Severity,
    EventSession, group_alerts_into_sessions,
)
from recommendations import (
    RecommendationEngine, Recommendation, Urgency, PILOT_RULES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(tag, detector, start_offset_h, duration_h=1, severity=Severity.HIGH,
                  value=82.0, limit=83.0):
    """Build a single EventSession with one synthetic alert."""
    start = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=start_offset_h)
    end   = start + pd.Timedelta(hours=duration_h)
    alert = Alert(
        timestamp=start,
        detector=detector,
        alert_type=AlertType.EWMA_VIOLATION,
        severity=severity,
        tag=tag,
        value=value,
        limit=limit,
        message=f"test alert {tag}",
    )
    sessions = group_alerts_into_sessions([alert], gap_minutes=30)
    return sessions[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecommendationEngineInit:
    def test_default_rules_loaded(self):
        re = RecommendationEngine()
        assert len(re.rules) == len(PILOT_RULES)

    def test_custom_rules(self):
        from recommendations import R01_ThermalDrift
        re = RecommendationEngine(rules=[R01_ThermalDrift()])
        assert len(re.rules) == 1


class TestGenerateEmptyInput:
    def test_empty_sessions_returns_empty(self, df_synthetic):
        re = RecommendationEngine()
        recs = re.generate([], df_synthetic)
        assert recs == []


class TestGenerateR01ThermalDrift:
    def test_r01_fires_on_high_T_R(self, df_synthetic):
        """R01 fires when T_R is high (>79.8) and detected by SPC.EWMA."""
        session = _make_session(
            tag="cstr.T_R_C",
            detector="SPC.EWMA",
            start_offset_h=2,
            value=82.0,
            limit=83.0,
        )
        # Ensure process data has high T_R
        df = df_synthetic.copy()
        df["cstr.T_R_C"] = 82.5
        df["cstr.F_cool"] = 0.30

        re = RecommendationEngine()
        recs = re.generate([session], df)
        r01_recs = [r for r in recs if r.rule_fired == "R01_ThermalDrift"]
        assert len(r01_recs) >= 1

    def test_r01_recommendation_structure(self, df_synthetic):
        session = _make_session("cstr.T_R_C", "SPC.EWMA", 2, value=82.5)
        df = df_synthetic.copy()
        df["cstr.T_R_C"] = 82.5
        df["cstr.F_cool"] = 0.30

        re = RecommendationEngine()
        recs = re.generate([session], df)
        r01_recs = [r for r in recs if r.rule_fired == "R01_ThermalDrift"]
        if r01_recs:
            rec = r01_recs[0]
            assert rec.action is not None
            assert rec.confidence > 0
            assert rec.urgency in list(Urgency)


class TestDeduplication:
    def test_same_rule_not_fired_twice_within_window(self, df_synthetic):
        """Two sessions close in time should produce only one recommendation per rule."""
        s1 = _make_session("cstr.T_R_C", "SPC.EWMA", 1, value=82.0)
        s2 = _make_session("cstr.T_R_C", "SPC.EWMA", 1.2, value=82.5)   # 12 min later
        df = df_synthetic.copy()
        df["cstr.T_R_C"] = 82.0
        df["cstr.F_cool"] = 0.30

        re = RecommendationEngine(dedup_window_minutes=60)
        recs = re.generate([s1, s2], df)
        r01_recs = [r for r in recs if r.rule_fired == "R01_ThermalDrift"]
        assert len(r01_recs) <= 1


class TestSortedByPriority:
    def test_recs_sorted_by_timestamp_then_priority(self, df_synthetic):
        sessions = [
            _make_session("cstr.T_R_C", "SPC.EWMA", h, value=82.0)
            for h in range(5)
        ]
        df = df_synthetic.copy()
        df["cstr.T_R_C"] = 82.0
        df["cstr.F_cool"] = 0.30

        re = RecommendationEngine()
        recs = re.generate(sessions, df)
        assert isinstance(recs, list)
        if len(recs) >= 2:
            sorted_recs = sorted(recs, key=lambda r: (r.timestamp, -r.priority_score))
            assert [r.id for r in recs] == [r.id for r in sorted_recs]


class TestGenerateToDataframe:
    def test_dataframe_has_columns(self, simple_sessions, df_synthetic):
        re = RecommendationEngine()
        df_recs = re.generate_to_dataframe(simple_sessions, df_synthetic)
        if len(df_recs) > 0:
            for col in ("id", "urgency", "confidence"):
                assert col in df_recs.columns
