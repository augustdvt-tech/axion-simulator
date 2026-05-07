"""
Axion AI - Outcome Tracker
==========================

Measures the actual outcome of executed actions against their predicted
impact. This is the closing of the learning loop: every executed action
produces evidence about whether the recommendation was correct, which
the system uses to:

    1. Adjust per-rule confidence over time (rules that work well gain
       confidence; rules that consistently miss lose it).

    2. Build up a database of "patterns of success/failure" that future
       phase-3 ML models will learn from.

    3. Provide the operator with a self-assessment: 'over the last 30 days,
       my recommendations were correct 87% of the time'.

How outcomes are computed
-------------------------
For each Recommendation that was executed, we look at the `expected_impact`
list. Each ExpectedImpact specifies:
    - which variable to measure
    - what value the recommendation predicted
    - how many minutes after execution to measure

The tracker waits for the measurement_delay to pass (in real time) or for
the corresponding row to appear in the historical CSV (in simulation mode),
then reads the actual variable value and computes:

    deviation_abs = |actual - predicted|
    deviation_pct = deviation_abs / |predicted - baseline|
    within_tolerance = deviation_pct < tolerance_threshold

The overall quality_score is the fraction of metrics within tolerance,
weighted by their operational importance.

Notes on simulation vs. production
----------------------------------
In production, OutcomeTracker runs as a background job: when a decision
is executed, it schedules an outcome measurement at decision_time +
measurement_delay. In our simulation framework, we already have the entire
process history loaded, so the measurement is just a lookup at the right
timestamp. This is faithful to the production design and only differs in
*how the data arrives*.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd

from recommendations import Recommendation, ExpectedImpact
from .models import Decision, Execution, Outcome, OutcomeMetric, ExecutionStatus


@dataclass
class OutcomeTrackerConfig:
    tolerance_pct: float = 0.50          # < 50% deviation => within tolerance
    default_measurement_delay_min: float = 30.0


class OutcomeTracker:
    """
    Computes Outcome records by looking up the actual values of impacted
    variables in the process data, at the timestamp specified by each
    ExpectedImpact's measurement_delay.

    Usage:
        tracker = OutcomeTracker()
        outcome = tracker.measure(rec, decision, execution, process_data)
    """

    def __init__(self, config: Optional[OutcomeTrackerConfig] = None):
        self.config = config or OutcomeTrackerConfig()

    def measure(
        self,
        rec: Recommendation,
        decision: Decision,
        execution: Execution,
        process_data: pd.DataFrame,
    ) -> Optional[Outcome]:
        """
        Measure the outcome of an executed action.

        Returns None if measurement is not possible:
        - decision was not accepted/modified
        - execution did not succeed
        - process data does not extend long enough past execution time
        - recommendation has no expected_impact entries
        """
        if execution.status != ExecutionStatus.SUCCESS and execution.status != ExecutionStatus.NOT_REQUIRED:
            return None
        if not rec.expected_impact:
            return None

        process_data = process_data.copy()
        process_data["timestamp"] = pd.to_datetime(process_data["timestamp"])
        last_ts = process_data["timestamp"].iloc[-1]

        metrics: List[OutcomeMetric] = []
        max_delay = 0.0
        any_measurable = False

        for imp in rec.expected_impact:
            delay = imp.time_to_effect_minutes or self.config.default_measurement_delay_min
            measurement_ts = execution.timestamp + pd.Timedelta(minutes=delay)
            max_delay = max(max_delay, delay)

            # If the data doesn't reach the measurement time, skip this metric
            if measurement_ts > last_ts:
                continue

            actual = self._lookup_value_at(process_data, imp.variable, measurement_ts)
            metric = OutcomeMetric(
                variable=imp.variable,
                predicted_value=imp.predicted_value,
                actual_value=actual,
                measurement_delay_minutes=delay,
            )

            # If a numeric prediction was made, evaluate against tolerance
            if metric.deviation_pct is not None:
                metric.within_tolerance = metric.deviation_pct < self.config.tolerance_pct
                any_measurable = True
            elif imp.predicted_value is None and actual is not None:
                # Predictions of "stable" or "informational" type — not scoreable
                # but should not penalize the rule. Mark as not-applicable.
                metric.within_tolerance = None

            metrics.append(metric)

        if not metrics:
            return None

        # Quality score: only computed from numerically scoreable metrics.
        # If a recommendation made no numeric predictions (e.g. INVESTIGATE
        # or WAIT_AND_MONITOR action), quality_score = None (no penalty).
        scoreable = [m for m in metrics if m.within_tolerance is not None]
        if scoreable:
            quality = sum(1.0 for m in scoreable if m.within_tolerance) / len(scoreable)
        else:
            # No scoreable metrics — recommendation was advisory only.
            # Default to neutral 0.5 so it doesn't bias rule confidence.
            quality = 0.5

        # Outcome timestamp = max measurement time
        outcome_ts = execution.timestamp + pd.Timedelta(minutes=max_delay)

        return Outcome(
            id=Outcome.new_id(),
            execution_id=execution.id,
            decision_id=decision.id,
            recommendation_id=rec.id,
            timestamp=outcome_ts,
            measurement_delay_minutes=max_delay,
            metrics=metrics,
            quality_score=quality,
            notes=self._summarize_outcome(metrics, quality),
        )

    @staticmethod
    def _lookup_value_at(
        df: pd.DataFrame, variable: str, ts: pd.Timestamp
    ) -> Optional[float]:
        """Return the value of `variable` at the row closest to `ts`."""
        if variable not in df.columns:
            return None
        sub = df[["timestamp", variable]].dropna()
        if sub.empty:
            return None
        # Closest row in time
        idx = (sub["timestamp"] - ts).abs().idxmin()
        return float(sub.loc[idx, variable])

    @staticmethod
    def _summarize_outcome(metrics: List[OutcomeMetric], quality: float) -> str:
        if not metrics:
            return "No measurable metrics"
        n_ok = sum(1 for m in metrics if m.within_tolerance)
        return f"{n_ok}/{len(metrics)} metrics within tolerance (quality={quality:.0%})"


# =============================================================================
# Per-rule performance tracker (the learning loop)
# =============================================================================

@dataclass
class RulePerformance:
    rule_name: str
    total_recommendations: int = 0
    accepted: int = 0
    modified: int = 0
    rejected: int = 0
    outcomes_measured: int = 0
    successes: int = 0                 # outcomes with quality >= 0.5
    failures: int = 0                  # outcomes with quality <  0.5

    @property
    def acceptance_rate(self) -> float:
        if self.total_recommendations == 0:
            return 0.0
        return (self.accepted + self.modified) / self.total_recommendations

    @property
    def success_rate(self) -> float:
        if self.outcomes_measured == 0:
            return 0.0
        return self.successes / self.outcomes_measured

    @property
    def confidence_adjustment(self) -> float:
        """
        Multiplier in [0.5, 1.5] derived from track record.
        Rules with high success rate and high acceptance rate get >1.0;
        rules consistently rejected or wrong get <1.0.
        Returns 1.0 (neutral) when there isn't enough data yet.
        """
        if self.outcomes_measured < 5 and self.total_recommendations < 10:
            return 1.0
        # Combine acceptance and success rates (each 0..1) into a multiplier
        # centered at 1.0
        a = self.acceptance_rate
        s = self.success_rate if self.outcomes_measured > 0 else 0.5
        score = 0.5 * a + 0.5 * s   # 0..1
        # Map [0..1] to [0.5..1.5]
        return 0.5 + score


class PerformanceTracker:
    """Tracks aggregate performance per rule. The Recommendation Engine
    can read these adjustments at runtime to bias confidence.

    Thread-unsafe. For the MVP this is fine; production version will need
    locking or use a backing database.
    """

    def __init__(self):
        self.by_rule: Dict[str, RulePerformance] = {}

    def record_recommendation(self, rec: Recommendation) -> None:
        rp = self.by_rule.setdefault(
            rec.rule_fired, RulePerformance(rule_name=rec.rule_fired)
        )
        rp.total_recommendations += 1

    def record_decision(self, rec: Recommendation, decision: Decision) -> None:
        rp = self.by_rule.setdefault(
            rec.rule_fired, RulePerformance(rule_name=rec.rule_fired)
        )
        if decision.status == DecisionStatus.ACCEPTED:
            rp.accepted += 1
        elif decision.status == DecisionStatus.MODIFIED:
            rp.modified += 1
        elif decision.status == DecisionStatus.REJECTED:
            rp.rejected += 1

    def record_outcome(self, rec: Recommendation, outcome: Outcome) -> None:
        rp = self.by_rule.setdefault(
            rec.rule_fired, RulePerformance(rule_name=rec.rule_fired)
        )
        rp.outcomes_measured += 1
        if outcome.quality_score >= 0.5:
            rp.successes += 1
        else:
            rp.failures += 1

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for rp in self.by_rule.values():
            rows.append({
                "rule":                rp.rule_name,
                "total_recs":          rp.total_recommendations,
                "accepted":            rp.accepted,
                "modified":            rp.modified,
                "rejected":            rp.rejected,
                "acceptance_rate":     rp.acceptance_rate,
                "outcomes_measured":   rp.outcomes_measured,
                "successes":           rp.successes,
                "failures":            rp.failures,
                "success_rate":        rp.success_rate,
                "confidence_adj":      rp.confidence_adjustment,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("total_recs", ascending=False).reset_index(drop=True)


# Required by record_decision signature
from .models import DecisionStatus
