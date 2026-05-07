"""Tests for analytics/sessions.py: EventSession and group_alerts_into_sessions."""

import pandas as pd
import pytest
from analytics import (
    Alert, AlertType, Severity,
    EventSession, group_alerts_into_sessions,
)


def make_alert(tag, minutes_offset, detector="SPC",
               alert_type=AlertType.EWMA_VIOLATION, severity=Severity.MEDIUM,
               value=80.0, limit=81.0):
    return Alert(
        timestamp=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=minutes_offset),
        detector=detector,
        alert_type=alert_type,
        severity=severity,
        tag=tag,
        value=value,
        limit=limit,
        message=f"alert at t+{minutes_offset}m",
    )


class TestGroupAlertsIntoSessions:
    def test_empty_input_returns_empty(self):
        assert group_alerts_into_sessions([]) == []

    def test_single_alert_creates_one_session(self):
        alert = make_alert("cstr.T_R_C", 0)
        sessions = group_alerts_into_sessions([alert])
        assert len(sessions) == 1

    def test_close_alerts_merged_into_one_session(self):
        alerts = [make_alert("cstr.T_R_C", t) for t in [0, 5, 10, 15]]
        sessions = group_alerts_into_sessions(alerts, gap_minutes=30)
        assert len(sessions) == 1

    def test_distant_alerts_create_separate_sessions(self):
        a1 = make_alert("cstr.T_R_C", 0)
        a2 = make_alert("cstr.T_R_C", 90)   # 90 min gap > default 30 min
        sessions = group_alerts_into_sessions([a1, a2], gap_minutes=30)
        assert len(sessions) == 2

    def test_different_tags_create_separate_sessions(self):
        a1 = make_alert("cstr.T_R_C", 0)
        a2 = make_alert("column.purity_B", 5)
        sessions = group_alerts_into_sessions([a1, a2])
        assert len(sessions) == 2

    def test_different_detectors_create_separate_sessions(self):
        a1 = make_alert("cstr.T_R_C", 0, detector="SPC")
        a2 = make_alert("cstr.T_R_C", 5, detector="PCA")
        sessions = group_alerts_into_sessions([a1, a2])
        assert len(sessions) == 2

    def test_peak_severity_is_max(self):
        alerts = [
            make_alert("cstr.T_R_C", 0, severity=Severity.LOW),
            make_alert("cstr.T_R_C", 5, severity=Severity.CRITICAL),
            make_alert("cstr.T_R_C", 10, severity=Severity.MEDIUM),
        ]
        sessions = group_alerts_into_sessions(alerts, gap_minutes=30)
        assert sessions[0].peak_severity == Severity.CRITICAL

    def test_session_duration(self):
        alerts = [make_alert("cstr.T_R_C", t) for t in [0, 10, 20]]
        sessions = group_alerts_into_sessions(alerts, gap_minutes=30)
        assert sessions[0].duration_minutes == pytest.approx(20.0)

    def test_alert_count(self):
        n = 5
        alerts = [make_alert("cstr.T_R_C", t * 5) for t in range(n)]
        sessions = group_alerts_into_sessions(alerts, gap_minutes=30)
        assert sessions[0].alert_count == n

    def test_session_to_row_has_required_keys(self):
        alert = make_alert("cstr.T_R_C", 0)
        session = group_alerts_into_sessions([alert])[0]
        row = session.to_row()
        for key in ("detector", "tag", "start_time", "end_time",
                    "duration_min", "alert_count", "peak_severity"):
            assert key in row

    def test_unsorted_input_is_handled(self):
        alerts = [
            make_alert("cstr.T_R_C", 20),
            make_alert("cstr.T_R_C", 5),
            make_alert("cstr.T_R_C", 0),
        ]
        sessions = group_alerts_into_sessions(alerts, gap_minutes=30)
        assert len(sessions) == 1
        assert sessions[0].alert_count == 3
