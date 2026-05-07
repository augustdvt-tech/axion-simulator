"""Tests for consensus/controller.py: ConsensusController."""

import pytest
import pandas as pd
from consensus import (
    ConsensusController, OperatingMode, DecisionStatus,
    RealisticOperator, ConservativeOperator, PILOT_SAFETY_LIMITS,
)
from recommendations import Recommendation, Action, ActionType, Urgency


def make_rec(urgency=Urgency.HIGH, target="column.RR",
             current=5.5, proposed=6.0, confidence=0.85):
    return Recommendation(
        id=Recommendation.new_id(),
        timestamp=pd.Timestamp("2026-01-01 02:00:00"),
        priority_score=0.75,
        diagnosis="Test diagnosis",
        probable_cause="Test cause",
        action=Action(
            type=ActionType.ADJUST_SETPOINT,
            description="Adjust RR",
            target_variable=target,
            current_value=current,
            proposed_value=proposed,
            adjustment=proposed - current,
            units="dimensionless",
        ),
        expected_impact=[],
        urgency=urgency,
        confidence=confidence,
    )


@pytest.fixture
def df_process(df_synthetic):
    return df_synthetic


class TestConsensusControllerAdvisorMode:
    def test_advisor_mode_processes_recs(self, df_process):
        # accept_critical_prob=1.0 ensures HIGH urgency recs are accepted
        operator = RealisticOperator(
            seed=0,
            accept_high_prob=1.0, accept_critical_prob=1.0,
            accept_medium_prob=1.0, accept_low_prob=1.0,
        )
        cc = ConsensusController(mode=OperatingMode.ADVISOR, operator=operator)
        rec = make_rec()
        log = cc.process([rec], df_process)
        assert log.summary()["recommendations"] == 1

    def test_advisor_mode_creates_decision(self, df_process):
        operator = RealisticOperator(
            seed=0,
            accept_high_prob=1.0, accept_critical_prob=1.0,
            accept_medium_prob=1.0, accept_low_prob=1.0,
        )
        cc = ConsensusController(mode=OperatingMode.ADVISOR, operator=operator)
        rec = make_rec()
        log = cc.process([rec], df_process)
        assert len(log.decisions) == 1

    def test_advisor_decision_is_accepted_or_rejected(self, df_process):
        operator = RealisticOperator(seed=1)
        cc = ConsensusController(mode=OperatingMode.ADVISOR, operator=operator)
        rec = make_rec()
        log = cc.process([rec], df_process)
        dec = log.decisions[0]
        assert dec.status in (DecisionStatus.ACCEPTED, DecisionStatus.MODIFIED,
                               DecisionStatus.REJECTED, DecisionStatus.DEFERRED)


class TestConsensusControllerSafetyGate:
    def test_out_of_range_action_rejected(self, df_process):
        """An action proposing a value outside safety limits should be auto-rejected."""
        from consensus import SafetyLimits
        limits = SafetyLimits(limits={"column.RR": {"min": 3.0, "max": 5.0}})
        operator = RealisticOperator(seed=0)
        cc = ConsensusController(
            mode=OperatingMode.ADVISOR,
            operator=operator,
            safety_limits=limits,
        )
        rec = make_rec(target="column.RR", current=5.5, proposed=8.0)  # 8.0 > max 5.0
        log = cc.process([rec], df_process)
        dec = log.decisions[0]
        assert dec.status == DecisionStatus.REJECTED

    def test_safe_action_not_rejected_by_safety(self, df_process):
        from consensus import SafetyLimits
        limits = SafetyLimits(limits={"column.RR": {"min": 3.0, "max": 8.0}})
        operator = RealisticOperator(seed=2)
        cc = ConsensusController(
            mode=OperatingMode.ADVISOR,
            operator=operator,
            safety_limits=limits,
        )
        rec = make_rec(target="column.RR", current=5.5, proposed=6.0)
        log = cc.process([rec], df_process)
        dec = log.decisions[0]
        assert dec.status != DecisionStatus.REJECTED or dec.justification != "safety"


class TestConsensusControllerConservativeOperator:
    def test_conservative_operator_rejects_more(self, df_process):
        cons = ConservativeOperator()
        real = RealisticOperator(seed=42)
        recs = [make_rec() for _ in range(20)]

        cc_cons = ConsensusController(mode=OperatingMode.ADVISOR, operator=cons)
        cc_real = ConsensusController(mode=OperatingMode.ADVISOR, operator=real)

        log_cons = cc_cons.process(recs, df_process)
        log_real = cc_real.process(recs, df_process)

        cons_accepts = sum(1 for d in log_cons.decisions if d.status == DecisionStatus.ACCEPTED)
        real_accepts = sum(1 for d in log_real.decisions if d.status == DecisionStatus.ACCEPTED)
        assert cons_accepts <= real_accepts


class TestDecisionLog:
    def test_summary_counts(self, df_process):
        operator = RealisticOperator(seed=0)
        cc = ConsensusController(mode=OperatingMode.ADVISOR, operator=operator)
        recs = [make_rec() for _ in range(3)]
        log = cc.process(recs, df_process)
        summary = log.summary()
        assert summary["recommendations"] == 3
        assert summary.get("decisions", 0) >= 0

    def test_get_recommendation(self, df_process):
        operator = RealisticOperator(seed=0)
        cc = ConsensusController(mode=OperatingMode.ADVISOR, operator=operator)
        rec = make_rec()
        log = cc.process([rec], df_process)
        retrieved = log.get_recommendation(rec.id)
        assert retrieved is not None
        assert retrieved.id == rec.id
