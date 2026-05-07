"""Tests for recommendations/rules_batch.py — batch reactor rule pack."""

import sys
from typing import Optional

import pandas as pd
import pytest

sys.path.insert(0, ".")

from analytics.alerts import Alert, AlertType, Severity
from analytics.sessions import EventSession
from recommendations.models import Urgency
from recommendations.rules_base import RuleContext
from recommendations.rules_batch import (
    BATCH_RULES, B01_HighReactorTemp, B02_RunawayRisk, B03_LowConversion,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alert(tag: str, value: float, severity: Severity = Severity.HIGH,
            ts: pd.Timestamp = pd.Timestamp("2026-01-01T00:00:00")) -> Alert:
    return Alert(
        timestamp=ts, detector="SPC.Shewhart",
        alert_type=AlertType.SHEWHART_VIOLATION, tag=tag,
        severity=severity, value=value, message="test",
    )


def _session(tag: str, peak: Severity = Severity.HIGH,
             alerts: Optional[list] = None,
             ts: pd.Timestamp = pd.Timestamp("2026-01-01T00:00:00")) -> EventSession:
    a = alerts or [_alert(tag, 100.0, peak, ts)]
    return EventSession(
        detector=a[0].detector, tag=tag,
        start_time=a[0].timestamp, end_time=a[-1].timestamp,
        peak_severity=peak, alert_count=len(a),
        first_alert=a[0], peak_alert=a[0],
    )


def _df_with(values: dict, n_rows: int = 30) -> pd.DataFrame:
    """Synthetic process_data with the given final-row values for each tag."""
    base = pd.Timestamp("2026-01-01")
    data = {"timestamp": [base + pd.Timedelta(minutes=i) for i in range(n_rows)]}
    for tag, v in values.items():
        data[tag] = [v] * n_rows
    return pd.DataFrame(data)


def _ctx(session: EventSession, df: pd.DataFrame,
         co_sessions: Optional[list] = None) -> RuleContext:
    return RuleContext(
        session_id=f"S-{session.tag}",
        process_data=df,
        co_occurring_sessions=co_sessions or [],
        operational_limits={},
    )


# ─────────────────────────────────────────────────────────────────────────────
# B01 — High Reactor Temperature
# =============================================================================

class TestB01HighReactorTemp:
    def test_matches_high_severity_on_T_R(self):
        rule = B01_HighReactorTemp()
        s = _session("batch.T_R_C", peak=Severity.HIGH)
        c = _ctx(s, _df_with({"batch.T_R_C": 96.0, "batch.F_cool": 0.4}))
        assert rule.matches(s, c) is True

    def test_does_not_match_low_severity(self):
        rule = B01_HighReactorTemp()
        s = _session("batch.T_R_C", peak=Severity.LOW)
        c = _ctx(s, _df_with({"batch.T_R_C": 80.0, "batch.F_cool": 0.4}))
        assert rule.matches(s, c) is False

    def test_does_not_match_other_tags(self):
        rule = B01_HighReactorTemp()
        s = _session("batch.dHdt", peak=Severity.HIGH)
        c = _ctx(s, _df_with({"batch.dHdt": 200.0}))
        assert rule.matches(s, c) is False

    def test_fire_returns_recommendation(self):
        rule = B01_HighReactorTemp()
        s = _session("batch.T_R_C", peak=Severity.HIGH)
        c = _ctx(s, _df_with({"batch.T_R_C": 96.0, "batch.F_cool": 0.4}))
        rec = rule.fire(s, c)
        assert rec is not None
        assert rec.rule_fired == "B01_HighReactorTemp"

    def test_fire_proposes_increased_coolant(self):
        rule = B01_HighReactorTemp()
        s = _session("batch.T_R_C", peak=Severity.HIGH)
        c = _ctx(s, _df_with({"batch.T_R_C": 96.0, "batch.F_cool": 0.4}))
        rec = rule.fire(s, c)
        assert rec.action.target_variable == "batch.F_cool"
        assert rec.action.proposed_value > rec.action.current_value


# =============================================================================
# B02 — Runaway Risk
# =============================================================================

class TestB02RunawayRisk:
    def test_does_not_match_without_co_occurring_T_R(self):
        rule = B02_RunawayRisk()
        s = _session("batch.dHdt", peak=Severity.CRITICAL)
        c = _ctx(s, _df_with({"batch.dHdt": 280.0, "batch.T_R_C": 90.0,
                                 "batch.F_cool": 0.4}))
        assert rule.matches(s, c) is False

    def test_matches_when_T_R_co_occurs(self):
        rule = B02_RunawayRisk()
        co = _session("batch.T_R_C", peak=Severity.HIGH)
        s  = _session("batch.dHdt", peak=Severity.CRITICAL)
        c = _ctx(s, _df_with({"batch.dHdt": 280.0, "batch.T_R_C": 92.0,
                                 "batch.F_cool": 0.4}), co_sessions=[co])
        assert rule.matches(s, c) is True

    def test_fire_urgency_critical(self):
        rule = B02_RunawayRisk()
        co = _session("batch.T_R_C", peak=Severity.HIGH)
        s  = _session("batch.dHdt", peak=Severity.CRITICAL)
        c = _ctx(s, _df_with({"batch.dHdt": 280.0, "batch.T_R_C": 92.0,
                                 "batch.F_cool": 0.4}), co_sessions=[co])
        rec = rule.fire(s, c)
        assert rec.urgency == Urgency.CRITICAL

    def test_fire_proposes_max_coolant(self):
        rule = B02_RunawayRisk()
        co = _session("batch.T_R_C", peak=Severity.HIGH)
        s  = _session("batch.dHdt", peak=Severity.CRITICAL)
        c = _ctx(s, _df_with({"batch.dHdt": 280.0, "batch.T_R_C": 92.0,
                                 "batch.F_cool": 0.4}), co_sessions=[co])
        rec = rule.fire(s, c)
        assert rec.action.proposed_value == 1.0


# =============================================================================
# B03 — Low Conversion
# =============================================================================

class TestB03LowConversion:
    def test_matches_medium_severity(self):
        rule = B03_LowConversion()
        s = _session("batch.conversion", peak=Severity.MEDIUM)
        c = _ctx(s, _df_with({"batch.conversion": 0.7}))
        assert rule.matches(s, c) is True

    def test_fire_action_is_investigate(self):
        from recommendations.models import ActionType
        rule = B03_LowConversion()
        s = _session("batch.conversion", peak=Severity.MEDIUM)
        c = _ctx(s, _df_with({"batch.conversion": 0.7}))
        rec = rule.fire(s, c)
        assert rec.action.type == ActionType.INVESTIGATE

    def test_fire_no_numeric_impact(self):
        rule = B03_LowConversion()
        s = _session("batch.conversion", peak=Severity.MEDIUM)
        c = _ctx(s, _df_with({"batch.conversion": 0.7}))
        rec = rule.fire(s, c)
        assert rec.expected_impact == []


# =============================================================================
# Pack
# =============================================================================

class TestBatchPack:
    def test_pack_has_three_rules(self):
        assert len(BATCH_RULES) == 3

    def test_rule_names_unique(self):
        names = [r.rule_name for r in BATCH_RULES]
        assert len(set(names)) == len(names)

    def test_rule_names_use_b_prefix(self):
        for r in BATCH_RULES:
            assert r.rule_name.startswith("B")
