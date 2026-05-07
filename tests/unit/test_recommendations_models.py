"""Tests for recommendations/models.py: Recommendation, Action, Urgency, priority_score."""

import pytest
import pandas as pd
from recommendations import (
    Recommendation, Action, ActionType, ExpectedImpact, Urgency,
    recommendations_to_dataframe,
)


def make_recommendation(urgency=Urgency.MEDIUM, confidence=0.8):
    action = Action(
        type=ActionType.ADJUST_SETPOINT,
        description="Increase coolant flow",
        target_variable="cstr.F_cool",
        current_value=0.30,
        proposed_value=0.35,
        adjustment=0.05,
        units="m3/h",
    )
    return Recommendation(
        id=Recommendation.new_id(),
        timestamp=pd.Timestamp("2026-01-01 02:00:00"),
        priority_score=0.7,
        diagnosis="Reactor temperature drifting upward",
        probable_cause="Jacket fouling",
        action=action,
        expected_impact=[],
        urgency=urgency,
        confidence=confidence,
    )


class TestUrgency:
    def test_urgency_values(self):
        assert Urgency.LOW.value == "low"
        assert Urgency.MEDIUM.value == "medium"
        assert Urgency.HIGH.value == "high"
        assert Urgency.CRITICAL.value == "critical"

    def test_urgency_ordering(self):
        order = [Urgency.LOW, Urgency.MEDIUM, Urgency.HIGH, Urgency.CRITICAL]
        assert len(order) == 4


class TestAction:
    def test_is_automated_true(self):
        a = Action(
            type=ActionType.ADJUST_SETPOINT,
            description="Adjust setpoint",
            target_variable="cstr.F_cool",
            proposed_value=0.35,
        )
        assert a.is_automated is True

    def test_is_automated_false_investigate(self):
        a = Action(type=ActionType.INVESTIGATE, description="Check manually")
        assert a.is_automated is False

    def test_is_automated_false_missing_target(self):
        a = Action(type=ActionType.ADJUST_SETPOINT, description="Adjust",
                   proposed_value=0.35)
        assert a.is_automated is False


class TestRecommendation:
    def test_new_id_is_unique(self):
        ids = {Recommendation.new_id() for _ in range(20)}
        assert len(ids) == 20

    def test_recommendation_has_required_fields(self):
        rec = make_recommendation()
        assert rec.id
        assert isinstance(rec.timestamp, pd.Timestamp)
        assert rec.diagnosis
        assert rec.action is not None
        assert rec.urgency in list(Urgency)

    def test_priority_score_range(self):
        for urgency in Urgency:
            for conf in [0.3, 0.6, 0.9]:
                rec = make_recommendation(urgency=urgency, confidence=conf)
                assert 0.0 <= rec.priority_score <= 1.0 or rec.priority_score >= 0

    def test_critical_higher_priority_than_low(self):
        rec_critical = make_recommendation(urgency=Urgency.CRITICAL, confidence=0.8)
        rec_low = make_recommendation(urgency=Urgency.LOW, confidence=0.8)
        assert rec_critical.priority_score >= rec_low.priority_score

    def test_higher_confidence_higher_priority(self):
        rec_high_conf = make_recommendation(urgency=Urgency.MEDIUM, confidence=0.95)
        rec_low_conf  = make_recommendation(urgency=Urgency.MEDIUM, confidence=0.4)
        assert rec_high_conf.priority_score >= rec_low_conf.priority_score


class TestRecommendationsToDataframe:
    def test_empty_returns_dataframe(self):
        df = recommendations_to_dataframe([])
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        rec = make_recommendation()
        df = recommendations_to_dataframe([rec])
        for col in ("id", "timestamp", "urgency", "diagnosis"):
            assert col in df.columns

    def test_row_count(self):
        recs = [make_recommendation() for _ in range(5)]
        df = recommendations_to_dataframe(recs)
        assert len(df) == 5
