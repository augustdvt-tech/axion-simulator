"""Tests for consensus/operator_outcomes.py — UI decision outcome tracking."""

import sys
from typing import List

import pandas as pd
import pytest

sys.path.insert(0, ".")

from consensus.models import (
    Decision, DecisionStatus, Execution, ExecutionStatus, Outcome,
)
from consensus.operator_outcomes import (
    OperatorOverride,
    is_outcome_measurable,
    measure_one,
    measure_pending,
    outcome_summary_dict,
    synthesize_decision_and_execution,
)
from recommendations.models import (
    Action, ActionType, ExpectedImpact, Recommendation, Urgency,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_rec(
    rec_id: str = "REC-A",
    rule: str = "R01",
    impacts: List[ExpectedImpact] | None = None,
    timestamp: pd.Timestamp | None = None,
) -> Recommendation:
    """Build a minimal Recommendation with an expected impact on column.purity_B."""
    if impacts is None:
        impacts = [ExpectedImpact(
            variable="column.purity_B",
            current_value=98.0,
            predicted_value=99.0,
            time_to_effect_minutes=30.0,
            description="Purity should rise after RR adjustment",
        )]
    return Recommendation(
        id=rec_id,
        timestamp=timestamp or pd.Timestamp("2026-01-01T00:00:00"),
        urgency=Urgency.MEDIUM,
        priority_score=0.5,
        confidence=0.7,
        diagnosis="test",
        probable_cause="test",
        rule_fired=rule,
        affected_variables=["column.purity_B"],
        action=Action(
            type=ActionType.ADJUST_SETPOINT,
            description="Increase RR",
            target_variable="column.RR",
            current_value=5.0,
            proposed_value=5.5,
            adjustment=0.5,
            units="ratio",
        ),
        expected_impact=impacts,
    )


def _process_df_holding_at(value: float, n_rows: int = 200, freq: str = "1min") -> pd.DataFrame:
    """Synthetic process data with column.purity_B held constant at `value`."""
    return pd.DataFrame({
        "timestamp":       pd.date_range("2026-01-01", periods=n_rows, freq=freq),
        "column.purity_B": [value] * n_rows,
    })


def _override(
    rec_id: str = "REC-A",
    status: str = "accepted",
    decision_ts: str = "2026-01-01T00:00:00",
) -> OperatorOverride:
    return OperatorOverride(
        rec_id=rec_id,
        status=status,
        justification="ok",
        decision_ts=pd.Timestamp(decision_ts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# is_outcome_measurable
# ─────────────────────────────────────────────────────────────────────────────

class TestIsOutcomeMeasurable:
    def test_rejected_decision_never_measurable(self):
        rec = _make_rec()
        ov = _override(status="rejected")
        now = pd.Timestamp("2026-01-01T05:00:00")
        assert is_outcome_measurable(ov, rec, now) is False

    def test_accepted_after_delay_is_measurable(self):
        rec = _make_rec()
        ov = _override(decision_ts="2026-01-01T00:00:00")
        now = pd.Timestamp("2026-01-01T00:30:00")   # exactly 30 min later
        assert is_outcome_measurable(ov, rec, now) is True

    def test_accepted_before_delay_not_measurable(self):
        rec = _make_rec()
        ov = _override(decision_ts="2026-01-01T00:00:00")
        now = pd.Timestamp("2026-01-01T00:15:00")
        assert is_outcome_measurable(ov, rec, now) is False

    def test_modified_decision_is_measurable(self):
        rec = _make_rec()
        ov = _override(status="modified", decision_ts="2026-01-01T00:00:00")
        now = pd.Timestamp("2026-01-01T01:00:00")
        assert is_outcome_measurable(ov, rec, now) is True

    def test_no_expected_impact_not_measurable(self):
        rec = _make_rec(impacts=[])
        ov = _override()
        now = pd.Timestamp("2026-01-01T01:00:00")
        assert is_outcome_measurable(ov, rec, now) is False

    def test_uses_max_delay_across_impacts(self):
        impacts = [
            ExpectedImpact(variable="column.purity_B", current_value=98.0,
                           predicted_value=99.0, time_to_effect_minutes=10.0),
            ExpectedImpact(variable="cstr.T_R_C", current_value=80.0,
                           predicted_value=78.0, time_to_effect_minutes=60.0),
        ]
        rec = _make_rec(impacts=impacts)
        ov  = _override(decision_ts="2026-01-01T00:00:00")
        # 30 min after — only the 10-min impact is "ready"
        assert is_outcome_measurable(ov, rec, pd.Timestamp("2026-01-01T00:30:00")) is False
        # 60 min after — both ready
        assert is_outcome_measurable(ov, rec, pd.Timestamp("2026-01-01T01:00:00")) is True


# ─────────────────────────────────────────────────────────────────────────────
# synthesize_decision_and_execution
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesize:
    def test_returns_decision_and_execution(self):
        rec = _make_rec()
        ov  = _override()
        d, e = synthesize_decision_and_execution(rec, ov)
        assert isinstance(d, Decision)
        assert isinstance(e, Execution)

    def test_decision_status_mapped_correctly(self):
        rec = _make_rec()
        cases = [
            ("accepted", DecisionStatus.ACCEPTED),
            ("rejected", DecisionStatus.REJECTED),
            ("modified", DecisionStatus.MODIFIED),
        ]
        for status_str, expected in cases:
            d, _ = synthesize_decision_and_execution(rec, _override(status=status_str))
            assert d.status == expected

    def test_execution_status_is_success(self):
        _, e = synthesize_decision_and_execution(_make_rec(), _override())
        assert e.status == ExecutionStatus.SUCCESS

    def test_decision_timestamps_match_override(self):
        ov = _override(decision_ts="2026-01-01T03:30:00")
        d, e = synthesize_decision_and_execution(_make_rec(), ov)
        assert d.timestamp == ov.decision_ts
        assert e.timestamp == ov.decision_ts


# ─────────────────────────────────────────────────────────────────────────────
# measure_one
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasureOne:
    def test_returns_outcome_when_data_extends_past_delay(self):
        rec = _make_rec()
        ov  = _override(decision_ts="2026-01-01T00:00:00")
        # Data extends 200 min, well past the 30-min delay
        df  = _process_df_holding_at(value=99.0, n_rows=200)
        outcome = measure_one(ov, rec, df)
        assert outcome is not None
        assert outcome.recommendation_id == rec.id

    def test_outcome_quality_high_when_actual_matches_prediction(self):
        rec = _make_rec()   # predicts purity_B = 99.0
        ov  = _override()
        df  = _process_df_holding_at(value=99.0)   # actual = 99.0 → perfect match
        outcome = measure_one(ov, rec, df)
        assert outcome.quality_score == pytest.approx(1.0)

    def test_outcome_quality_low_when_actual_far_from_prediction(self):
        rec = _make_rec()                       # predicts 99.0
        ov  = _override()
        # Tolerance defaults to 50% deviation; pick a value well past that.
        # deviation_pct = |99 - 10| / 99 = 0.90 → outside tolerance
        df  = _process_df_holding_at(value=10.0)
        outcome = measure_one(ov, rec, df)
        assert outcome.quality_score < 0.5

    def test_returns_none_when_data_too_short(self):
        rec = _make_rec()
        ov  = _override(decision_ts="2026-01-01T00:00:00")
        df  = _process_df_holding_at(value=99.0, n_rows=15)   # only 15 min
        # Need >30 min for the impact's measurement_delay; should yield None
        outcome = measure_one(ov, rec, df)
        assert outcome is None


# ─────────────────────────────────────────────────────────────────────────────
# measure_pending
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasurePending:
    def test_skips_already_measured(self):
        rec = _make_rec()
        ov  = _override()
        df  = _process_df_holding_at(value=99.0)
        out = measure_pending(
            overrides={"REC-A": ov},
            recs_by_id={"REC-A": rec},
            now_ts=pd.Timestamp("2026-01-01T01:00:00"),
            process_data=df,
            already_measured={"REC-A"},
        )
        assert out == []

    def test_skips_not_yet_due(self):
        rec = _make_rec()
        ov  = _override(decision_ts="2026-01-01T00:00:00")
        df  = _process_df_holding_at(value=99.0)
        out = measure_pending(
            overrides={"REC-A": ov},
            recs_by_id={"REC-A": rec},
            now_ts=pd.Timestamp("2026-01-01T00:10:00"),   # not yet 30 min
            process_data=df,
            already_measured=set(),
        )
        assert out == []

    def test_measures_due_outcomes(self):
        rec = _make_rec()
        ov  = _override(decision_ts="2026-01-01T00:00:00")
        df  = _process_df_holding_at(value=99.0)
        out = measure_pending(
            overrides={"REC-A": ov},
            recs_by_id={"REC-A": rec},
            now_ts=pd.Timestamp("2026-01-01T00:30:00"),
            process_data=df,
            already_measured=set(),
        )
        assert len(out) == 1
        assert out[0].recommendation_id == "REC-A"

    def test_skips_unknown_rec_ids(self):
        ov = _override(rec_id="REC-MISSING")
        df = _process_df_holding_at(value=99.0)
        out = measure_pending(
            overrides={"REC-MISSING": ov},
            recs_by_id={},   # empty — rec not registered
            now_ts=pd.Timestamp("2026-01-01T01:00:00"),
            process_data=df,
            already_measured=set(),
        )
        assert out == []

    def test_skips_rejected_overrides(self):
        rec = _make_rec()
        ov  = _override(status="rejected")
        df  = _process_df_holding_at(value=99.0)
        out = measure_pending(
            overrides={"REC-A": ov},
            recs_by_id={"REC-A": rec},
            now_ts=pd.Timestamp("2026-01-01T01:00:00"),
            process_data=df,
            already_measured=set(),
        )
        assert out == []


# ─────────────────────────────────────────────────────────────────────────────
# outcome_summary_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestOutcomeSummaryDict:
    def test_includes_required_keys(self):
        rec = _make_rec()
        ov  = _override()
        df  = _process_df_holding_at(value=99.0)
        outcome = measure_one(ov, rec, df)
        s = outcome_summary_dict(outcome, rec)
        for key in ("outcome_id", "rec_id", "rule_fired", "urgency",
                    "measured_at", "quality_score", "metrics"):
            assert key in s, f"Missing key: {key}"

    def test_metrics_are_listed(self):
        rec = _make_rec()
        ov  = _override()
        df  = _process_df_holding_at(value=99.0)
        outcome = measure_one(ov, rec, df)
        s = outcome_summary_dict(outcome, rec)
        assert isinstance(s["metrics"], list)
        assert len(s["metrics"]) >= 1
        m0 = s["metrics"][0]
        assert "variable" in m0 and "predicted_value" in m0 and "actual_value" in m0


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint /api/outcomes/operator
# ─────────────────────────────────────────────────────────────────────────────

class TestOperatorOutcomesEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_empty_when_none_measured(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "operator_outcomes", {})
        body = client.get("/api/outcomes/operator").json()
        assert body["count"] == 0
        assert body["outcomes"] == []

    def test_returns_measured_outcomes(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "operator_outcomes", {
            "REC-A": {
                "outcome_id":          "OUT-1",
                "rec_id":              "REC-A",
                "rule_fired":          "R01",
                "urgency":             "high",
                "measured_at":         "2026-01-01T00:30:00",
                "measurement_delay_min": 30.0,
                "quality_score":       0.85,
                "notes":               "1/1 within tolerance",
                "metrics":             [],
            },
        })
        body = client.get("/api/outcomes/operator").json()
        assert body["count"] == 1
        assert body["outcomes"][0]["rec_id"] == "REC-A"

    def test_sorted_descending_by_measured_at(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "operator_outcomes", {
            "REC-A": {"outcome_id": "OUT-1", "rec_id": "A",
                       "measured_at": "2026-01-01T00:00:00",
                       "quality_score": 0.5, "metrics": [],
                       "rule_fired": "R", "urgency": "low",
                       "measurement_delay_min": 30, "notes": ""},
            "REC-B": {"outcome_id": "OUT-2", "rec_id": "B",
                       "measured_at": "2026-01-01T05:00:00",
                       "quality_score": 0.5, "metrics": [],
                       "rule_fired": "R", "urgency": "low",
                       "measurement_delay_min": 30, "notes": ""},
        })
        body = client.get("/api/outcomes/operator").json()
        assert body["outcomes"][0]["rec_id"] == "B"
        assert body["outcomes"][1]["rec_id"] == "A"
