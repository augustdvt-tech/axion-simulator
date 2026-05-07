"""Tests for api/report.py — pure summary functions and render_html."""

import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, ".")

from api.report import (
    kpi_summary,
    recommendations_summary,
    decisions_summary,
    sessions_summary,
    render_html,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n=10, purity=99.0, T_R_C=78.0, include_ts=True) -> pd.DataFrame:
    base = datetime(2026, 1, 1, 0, 0, 0)
    data: dict = {
        "column.purity_B": [purity] * n,
        "cstr.T_R_C":      [T_R_C]  * n,
    }
    if include_ts:
        data["timestamp"] = [base + timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame(data)


def _rec(urgency="medium", rule="R01", status="pending") -> dict:
    return {
        "urgency":    urgency,
        "rule_fired": rule,
        "status":     status,
        "timestamp":  "2026-01-01T00:10:00",
        "diagnosis":  "test diagnosis",
    }


def _dec(status="accepted", urgency="high", rule_id="R01", ts="2026-01-01T00:15:00") -> dict:
    return {
        "status":        status,
        "justification": "ok",
        "timestamp":     ts,
        "urgency":       urgency,
        "rule_id":       rule_id,
    }


def _sess(detector="SPC.EWMA", tag="cstr.T_R_C", duration=5.0, severity="high") -> dict:
    return {
        "detector":      detector,
        "tag":           tag,
        "duration_min":  duration,
        "peak_severity": severity,
        "start_time":    "2026-01-01T00:05:00",
    }


def _render_minimal() -> str:
    df = _make_df()
    return render_html(
        scenario="test",
        generated_at="2026-01-01T00:00:00",
        kpi=kpi_summary(df),
        rec_summary=recommendations_summary([]),
        dec_summary=decisions_summary([]),
        sess_summary=sessions_summary([]),
        perf_rows=[],
        rec_log=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# kpi_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestKpiSummary:
    def test_empty_df_returns_zero_samples(self):
        r = kpi_summary(pd.DataFrame())
        assert r["n_samples"] == 0
        assert r["rows"] == []

    def test_counts_samples(self):
        r = kpi_summary(_make_df(n=20))
        assert r["n_samples"] == 20

    def test_duration_from_timestamps(self):
        df = _make_df(n=61)   # 61 rows × 1 min = 60 min = 1 h
        r = kpi_summary(df)
        assert abs(r["duration_h"] - 1.0) < 0.02

    def test_no_timestamps_duration_zero(self):
        r = kpi_summary(_make_df(include_ts=False))
        assert r["duration_h"] == 0.0

    def test_purity_all_above_spec(self):
        r = kpi_summary(_make_df(purity=99.5))
        assert r["purity_below_spec_pct"] == pytest.approx(0.0)

    def test_purity_all_below_spec(self):
        r = kpi_summary(_make_df(purity=97.0))
        assert r["purity_below_spec_pct"] == pytest.approx(100.0)

    def test_purity_half_below_spec(self):
        df = pd.DataFrame({
            "column.purity_B": [99.0, 99.0, 98.0, 98.0],
            "timestamp": [datetime(2026,1,1) + timedelta(minutes=i) for i in range(4)],
        })
        r = kpi_summary(df)
        assert r["purity_below_spec_pct"] == pytest.approx(50.0)

    def test_spec_max_violation_flagged(self):
        df = _make_df(T_R_C=90.0)   # cstr.T_R_C spec_max = 82
        r = kpi_summary(df)
        t_row = next(row for row in r["rows"] if row["col"] == "cstr.T_R_C")
        assert t_row["spec_violated"] is True

    def test_no_violation_when_within_spec(self):
        df = _make_df(T_R_C=78.0)   # below 82
        r = kpi_summary(df)
        t_row = next(row for row in r["rows"] if row["col"] == "cstr.T_R_C")
        assert t_row["spec_violated"] is False

    def test_kpi_row_has_expected_keys(self):
        r = kpi_summary(_make_df())
        row = r["rows"][0]
        for key in ("col", "label", "unit", "mean", "min", "max", "std", "spec_violated"):
            assert key in row, f"Missing key: {key}"

    def test_missing_columns_skipped(self):
        df = pd.DataFrame({"not_a_kpi_col": [1, 2, 3]})
        r = kpi_summary(df)
        assert r["rows"] == []


# ─────────────────────────────────────────────────────────────────────────────
# recommendations_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationsSummary:
    def test_empty_list(self):
        r = recommendations_summary([])
        assert r["total"] == 0
        assert r["acceptance_rate"] is None

    def test_total_count(self):
        r = recommendations_summary([_rec(), _rec(), _rec()])
        assert r["total"] == 3

    def test_all_pending_no_acceptance_rate(self):
        r = recommendations_summary([_rec("high"), _rec("low")])
        assert r["acceptance_rate"] is None
        assert r["n_decided"] == 0

    def test_acceptance_rate_all_accepted(self):
        recs = [_rec(status="accepted"), _rec(status="accepted")]
        r = recommendations_summary(recs)
        assert r["acceptance_rate"] == pytest.approx(1.0)

    def test_acceptance_rate_mixed(self):
        recs = [_rec(status="accepted"), _rec(status="rejected"), _rec()]
        r = recommendations_summary(recs)
        # 1 accepted / 2 decided
        assert r["acceptance_rate"] == pytest.approx(0.5)
        assert r["n_accepted"] == 1
        assert r["n_rejected"] == 1

    def test_by_urgency_counts(self):
        recs = [_rec("critical"), _rec("high"), _rec("high"), _rec("low")]
        r = recommendations_summary(recs)
        assert r["by_urgency"]["critical"] == 1
        assert r["by_urgency"]["high"] == 2
        assert r["by_urgency"]["low"] == 1

    def test_by_rule_counts(self):
        recs = [_rec(rule="R01"), _rec(rule="R01"), _rec(rule="R03")]
        r = recommendations_summary(recs)
        assert r["by_rule"]["R01"] == 2
        assert r["by_rule"]["R03"] == 1

    def test_n_decided_excludes_pending(self):
        recs = [_rec(status="accepted"), _rec(status="pending"), _rec(status="rejected")]
        r = recommendations_summary(recs)
        assert r["n_decided"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# decisions_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionsSummary:
    def test_empty(self):
        r = decisions_summary([])
        assert r["total"] == 0
        assert r["log"] == []

    def test_total(self):
        r = decisions_summary([_dec(), _dec(status="rejected")])
        assert r["total"] == 2

    def test_by_status_counter(self):
        decs = [_dec("accepted"), _dec("accepted"), _dec("rejected")]
        r = decisions_summary(decs)
        assert r["by_status"]["accepted"] == 2
        assert r["by_status"]["rejected"] == 1

    def test_log_sorted_desc(self):
        decs = [
            _dec(ts="2026-01-01T00:10:00"),
            _dec(ts="2026-01-01T00:05:00"),
            _dec(ts="2026-01-01T00:15:00"),
        ]
        r = decisions_summary(decs)
        timestamps = [d["timestamp"] for d in r["log"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_log_contains_all_fields(self):
        r = decisions_summary([_dec()])
        entry = r["log"][0]
        for key in ("status", "justification", "timestamp", "urgency", "rule_id"):
            assert key in entry


# ─────────────────────────────────────────────────────────────────────────────
# sessions_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionsSummary:
    def test_empty(self):
        r = sessions_summary([])
        assert r["total"] == 0
        assert r["longest"] == []

    def test_total(self):
        r = sessions_summary([_sess(), _sess()])
        assert r["total"] == 2

    def test_by_severity_counter(self):
        sessions = [_sess(severity="high"), _sess(severity="high"), _sess(severity="low")]
        r = sessions_summary(sessions)
        assert r["by_severity"]["high"] == 2
        assert r["by_severity"]["low"] == 1

    def test_by_detector_counter(self):
        sessions = [_sess(detector="SPC.CUSUM"), _sess(detector="PCA"), _sess(detector="SPC.CUSUM")]
        r = sessions_summary(sessions)
        assert r["by_detector"]["SPC.CUSUM"] == 2

    def test_longest_top5(self):
        sessions = [_sess(duration=float(i)) for i in range(10, 0, -1)]
        r = sessions_summary(sessions)
        durations = [s["duration_min"] for s in r["longest"]]
        assert len(durations) <= 5
        assert durations == sorted(durations, reverse=True)

    def test_longest_fewer_than_5(self):
        sessions = [_sess(duration=3.0), _sess(duration=7.0)]
        r = sessions_summary(sessions)
        assert len(r["longest"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# render_html
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderHtml:
    def test_returns_string(self):
        assert isinstance(_render_minimal(), str)

    def test_contains_doctype(self):
        assert "<!DOCTYPE html>" in _render_minimal()

    def test_contains_scenario_name(self):
        df = _make_df()
        html = render_html(
            scenario="thermal_drift",
            generated_at="2026-01-01T00:00:00",
            kpi=kpi_summary(df),
            rec_summary=recommendations_summary([]),
            dec_summary=decisions_summary([]),
            sess_summary=sessions_summary([]),
            perf_rows=[],
            rec_log=[],
        )
        assert "thermal_drift" in html

    def test_no_external_urls(self):
        html = _render_minimal()
        # self-contained report should not load external resources
        assert "https://" not in html
        assert "http://" not in html

    def test_print_media_query_present(self):
        assert "@media print" in _render_minimal()

    def test_executive_tiles_present(self):
        html = _render_minimal()
        assert "tile-grid" in html
        assert "Avg. Product Purity" in html

    def test_rec_urgency_badge_rendered(self):
        recs = [_rec("critical", status="accepted")]
        df = _make_df()
        html = render_html(
            scenario="s",
            generated_at="2026-01-01T00:00:00",
            kpi=kpi_summary(df),
            rec_summary=recommendations_summary(recs),
            dec_summary=decisions_summary([]),
            sess_summary=sessions_summary([]),
            perf_rows=[],
            rec_log=recs,
        )
        assert "CRITICAL" in html

    def test_spec_violation_class_in_kpi_table(self):
        df = _make_df(T_R_C=90.0)  # violates spec_max=82
        html = render_html(
            scenario="s",
            generated_at="2026-01-01T00:00:00",
            kpi=kpi_summary(df),
            rec_summary=recommendations_summary([]),
            dec_summary=decisions_summary([]),
            sess_summary=sessions_summary([]),
            perf_rows=[],
            rec_log=[],
        )
        assert "spec-viol" in html

    def test_perf_table_rendered(self):
        perf_rows = [{"rule_id": "R01", "n_issued": 3, "n_accepted": 2,
                      "n_rejected": 1, "acceptance_rate": 0.667}]
        df = _make_df()
        html = render_html(
            scenario="s",
            generated_at="2026-01-01T00:00:00",
            kpi=kpi_summary(df),
            rec_summary=recommendations_summary([]),
            dec_summary=decisions_summary([]),
            sess_summary=sessions_summary([]),
            perf_rows=perf_rows,
            rec_log=[],
        )
        assert "R01" in html
        assert "67%" in html


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint — state.run is None
# ─────────────────────────────────────────────────────────────────────────────

class TestReportEndpoint:
    @pytest.fixture
    def client(self):
        from api.server import app
        from fastapi.testclient import TestClient
        return TestClient(app, raise_server_exceptions=False)

    def test_404_when_no_scenario(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "run", None)
        r = client.get("/api/report/current")
        assert r.status_code == 404

    @pytest.fixture
    def mock_run(self, monkeypatch):
        from unittest.mock import MagicMock
        from api import server
        run = MagicMock()
        run.scenario = "test_scenario"
        run.process_data = _make_df()
        run.recommendations = []
        run.decisions = []
        run.sessions = []
        run.performance.summary_dataframe.return_value = pd.DataFrame()
        monkeypatch.setattr(server.state, "run", run)
        monkeypatch.setattr(server.state, "operator_overrides", {})
        return run

    def test_returns_html_content_type(self, client, mock_run):
        r = client.get("/api/report/current")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_html_body_contains_axion(self, client, mock_run):
        r = client.get("/api/report/current")
        assert r.status_code == 200
        assert "AXION" in r.text
