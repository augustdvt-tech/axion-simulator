"""
Axion AI - Recommendation Engine
================================

The orchestrator for the recommendation layer. Takes event sessions produced
by the Analytical Engine, evaluates each against the rule library, and
produces a list of Recommendations sorted by priority.

Core behaviors:

1. DEDUPLICATION. A single event (e.g. a thermal drift) may trigger multiple
   rules simultaneously. The engine deduplicates by rule name: a rule that
   already fired for a given event does not fire again until a cooldown
   period has passed.

2. CO-OCCURRENCE CONTEXT. Rules need to know what OTHER sessions are active
   around the same time (see Rule R02_FeedComposition: it only fires when
   PCA.T2 AND C_A AND purity sessions coexist). The engine builds a
   co-occurrence window for each session before evaluating rules.

3. PRIORITIZATION. Recommendations are sorted by priority_score so that the
   UI can show the most urgent ones at the top of the list.

4. STATELESSNESS. The engine itself has no state beyond the rule set. Same
   input (sessions + data) always produces the same output. This is
   important for testing and auditability.

The engine's output is the single contract shared with the future UI and
consensus module: a list of Recommendation objects.
"""

from __future__ import annotations
from typing import List, Optional, Dict
import pandas as pd

from analytics import EventSession
from .models import Recommendation, recommendations_to_dataframe
from .rules_base import DiagnosticRule, RuleContext
from .rules_pilot import PILOT_RULES


# Operational limits used by rules — passed to them via RuleContext
# Source: same limits used by the TrendDetector, centralized here for clarity
from analytics import PILOT_OPERATIONAL_LIMITS


class RecommendationEngine:
    """
    Evaluates rules against event sessions and produces recommendations.

    Usage:
        engine = RecommendationEngine()
        recs = engine.generate(sessions, process_data)
    """

    def __init__(
        self,
        rules: Optional[List[DiagnosticRule]] = None,
        operational_limits: Optional[Dict] = None,
        co_occurrence_minutes: float = 90.0,
        dedup_window_minutes: float = 60.0,
    ):
        self.rules = rules if rules is not None else list(PILOT_RULES)
        self.operational_limits = operational_limits or PILOT_OPERATIONAL_LIMITS
        self.co_occurrence_minutes = co_occurrence_minutes
        self.dedup_window_minutes = dedup_window_minutes

    # ---- public API ----

    def generate(
        self,
        sessions: List[EventSession],
        process_data: pd.DataFrame,
    ) -> List[Recommendation]:
        """Produce recommendations from a list of sessions."""
        if not sessions:
            return []

        # Prepare timestamps
        process_data = process_data.copy()
        process_data["timestamp"] = pd.to_datetime(process_data["timestamp"])

        # Sort sessions chronologically
        sessions = sorted(sessions, key=lambda s: s.start_time)

        recommendations: List[Recommendation] = []
        # Track: (rule_name, key) -> last timestamp, for dedup
        last_fired: Dict[tuple, pd.Timestamp] = {}
        dedup_window = pd.Timedelta(minutes=self.dedup_window_minutes)
        co_window = pd.Timedelta(minutes=self.co_occurrence_minutes)

        for idx, session in enumerate(sessions):
            # Build the context for this session
            co_occurring = self._find_co_occurring(session, sessions, co_window)
            # Slice the process data up to the session start time. Rules should
            # reason about the state of the process at the moment the event
            # began, not the future — otherwise they get to 'cheat' by
            # knowing what happens after the event.
            slice_mask = process_data["timestamp"] <= session.start_time
            sliced_data = process_data.loc[slice_mask]

            # Synthetic session id for the audit trail
            session_id = f"SES-{idx:05d}"

            context = RuleContext(
                process_data=sliced_data,
                operational_limits=self.operational_limits,
                co_occurring_sessions=co_occurring,
                session_id=session_id,
            )

            # Evaluate all rules
            for rule in self.rules:
                try:
                    if not rule.matches(session, context):
                        continue
                except Exception:
                    # A broken rule should not break the engine. Skip.
                    continue

                # Dedup key: rule + the dominant tag (or 'multi' for multivariate)
                tag_key = session.tag or "multi"
                dedup_key = (rule.rule_name, tag_key)

                last_ts = last_fired.get(dedup_key)
                if last_ts is not None and (session.start_time - last_ts) < dedup_window:
                    continue

                try:
                    rec = rule.fire(session, context)
                except Exception:
                    continue
                if rec is None:
                    continue

                recommendations.append(rec)
                last_fired[dedup_key] = session.start_time

        # Sort by timestamp then by priority (higher priority first for ties)
        recommendations.sort(
            key=lambda r: (r.timestamp, -r.priority_score)
        )
        return recommendations

    def generate_to_dataframe(
        self,
        sessions: List[EventSession],
        process_data: pd.DataFrame,
    ) -> pd.DataFrame:
        return recommendations_to_dataframe(
            self.generate(sessions, process_data)
        )

    # ---- helpers ----

    def _find_co_occurring(
        self,
        session: EventSession,
        all_sessions: List[EventSession],
        window: pd.Timedelta,
    ) -> List[EventSession]:
        """
        Return all sessions that overlap with `session` within the
        co-occurrence window, excluding the session itself.
        """
        result = []
        for other in all_sessions:
            if other is session:
                continue
            # Consider 'co-occurring' if start times are within `window`
            # OR if the sessions actually overlap in time
            start_diff = abs((other.start_time - session.start_time).total_seconds())
            if start_diff <= window.total_seconds():
                result.append(other)
                continue
            # Overlap check
            if (other.start_time <= session.end_time
                and other.end_time >= session.start_time):
                result.append(other)
        return result
