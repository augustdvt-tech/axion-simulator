"""Tests for consensus/models.py: Decision, DecisionStatus, Outcome."""

import pytest
import pandas as pd
from consensus import (
    Decision, DecisionStatus, Execution, ExecutionStatus,
    Outcome, OutcomeMetric,
    decisions_to_dataframe, outcomes_to_dataframe,
)
from recommendations import Recommendation, Action, ActionType, Urgency


def make_rec():
    return Recommendation(
        id=Recommendation.new_id(),
        timestamp=pd.Timestamp("2026-01-01 01:00:00"),
        priority_score=0.7,
        diagnosis="Test diagnosis",
        probable_cause="Test cause",
        action=Action(
            type=ActionType.ADJUST_SETPOINT,
            description="Increase RR",
            target_variable="column.RR",
            current_value=5.5,
            proposed_value=6.0,
            adjustment=0.5,
            units="dimensionless",
        ),
        expected_impact=[],
        urgency=Urgency.MEDIUM,
        confidence=0.80,
    )


class TestDecisionStatus:
    def test_all_statuses_defined(self):
        statuses = list(DecisionStatus)
        assert DecisionStatus.PENDING in statuses
        assert DecisionStatus.ACCEPTED in statuses
        assert DecisionStatus.REJECTED in statuses
        assert DecisionStatus.MODIFIED in statuses


class TestDecision:
    def test_create_accepted_decision(self):
        rec = make_rec()
        dec = Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            status=DecisionStatus.ACCEPTED,
            timestamp=pd.Timestamp("2026-01-01 01:05:00"),
            operator_id="test",
        )
        assert dec.status == DecisionStatus.ACCEPTED
        assert dec.recommendation_id == rec.id

    def test_create_rejected_decision(self):
        rec = make_rec()
        dec = Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            status=DecisionStatus.REJECTED,
            justification="Not relevant now",
            timestamp=pd.Timestamp("2026-01-01 01:05:00"),
            operator_id="test",
        )
        assert dec.status == DecisionStatus.REJECTED
        assert dec.justification == "Not relevant now"

    def test_new_id_unique(self):
        ids = {Decision.new_id() for _ in range(10)}
        assert len(ids) == 10

    def test_modified_decision_has_actual_action(self):
        rec = make_rec()
        modified_action = Action(
            type=ActionType.ADJUST_SETPOINT,
            description="Custom adjustment",
            target_variable="column.RR",
            current_value=5.5,
            proposed_value=5.8,
        )
        dec = Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            status=DecisionStatus.MODIFIED,
            actual_action=modified_action,
            timestamp=pd.Timestamp("2026-01-01 01:05:00"),
            operator_id="test",
        )
        assert dec.actual_action.proposed_value == 5.8


class TestDecisionsToDataframe:
    def test_empty_returns_dataframe(self):
        df = decisions_to_dataframe([])
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        rec = make_rec()
        dec = Decision(
            id=Decision.new_id(),
            recommendation_id=rec.id,
            status=DecisionStatus.ACCEPTED,
            timestamp=pd.Timestamp("2026-01-01 01:05:00"),
            operator_id="test",
        )
        df = decisions_to_dataframe([dec])
        for col in ("id", "recommendation_id", "status"):
            assert col in df.columns

    def test_row_count(self):
        rec = make_rec()
        decs = [
            Decision(
                id=Decision.new_id(),
                recommendation_id=rec.id,
                status=DecisionStatus.ACCEPTED,
                timestamp=pd.Timestamp("2026-01-01"),
                operator_id="test",
            )
            for _ in range(3)
        ]
        df = decisions_to_dataframe(decs)
        assert len(df) == 3


class TestOutcome:
    def test_create_outcome(self):
        metric = OutcomeMetric(
            variable="cstr.T_R_C",
            predicted_value=-1.5,
            actual_value=-1.2,
            measurement_delay_minutes=15.0,
        )
        outcome = Outcome(
            id=Outcome.new_id(),
            execution_id="EXE-001",
            decision_id="dec-001",
            recommendation_id="REC-001",
            timestamp=pd.Timestamp("2026-01-01 02:00:00"),
            measurement_delay_minutes=15.0,
            metrics=[metric],
        )
        assert outcome.decision_id == "dec-001"
        assert len(outcome.metrics) == 1
        assert outcome.metrics[0].variable == "cstr.T_R_C"
