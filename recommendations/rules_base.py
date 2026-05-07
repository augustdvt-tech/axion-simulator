"""
Axion AI - Rule Base (Knowledge Representation)
===============================================

A DiagnosticRule is a piece of process knowledge captured as code:

    - matches(session, context) -> bool         # does this rule apply?
    - fire(session, context)    -> Recommendation # produce the recommendation

This explicit representation is a deliberate design choice for the MVP
(and for the early operational phase of Axion AI):

1. TRACEABILITY. When a recommendation fires, the operator can see *exactly*
   which rule was triggered and why. This is what makes the system auditable
   and what lets engineers trust its suggestions. A neural network can't
   explain to a shift operator why it's recommending +0.3 on reflux ratio;
   a rule named `LowPurityHighConversion` can.

2. EVOLVABILITY. Rules are isolated. Adding a new rule for a new failure
   mode = adding one class. Modifying a rule's threshold = changing one number.
   The recommendation logic is not a black box that has to be retrained.

3. EXPERTISE CAPTURE. Rules are written in the language process engineers
   use when they diagnose problems on the plant floor. This makes it easy
   to review rules with domain experts, add new ones based on post-event
   analysis, and transfer knowledge from senior to junior engineers through
   the system itself.

Over time, the rule base is supplemented by ML-based diagnostics
(case-based reasoning from historical decisions). That's Phase 3, not MVP.

--- Rule Context ---

A rule needs more than just the EventSession to decide: it needs the
current values of process variables, the recent history, and the
operational limits. The `RuleContext` bundles all of this.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

from analytics import EventSession
from .models import Recommendation


@dataclass
class RuleContext:
    """
    Everything a rule needs to produce a recommendation beyond the
    EventSession itself.
    """
    # The entire process data up to the current moment
    process_data: pd.DataFrame
    # Operational limits (same structure as used by TrendDetector)
    operational_limits: Dict[str, Dict[str, Optional[float]]]
    # All sessions active around the time of the triggering session
    # (within session_window_minutes). Lets rules that need to combine
    # multiple evidence streams work correctly.
    co_occurring_sessions: List[EventSession]
    # Session ID map (session -> synthetic identifier)
    session_id: str

    def current_value(self, tag: str) -> Optional[float]:
        """Value of a tag at (or just before) the triggering timestamp."""
        if tag not in self.process_data.columns:
            return None
        series = self.process_data[tag].dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])

    def value_at(self, tag: str, timestamp: pd.Timestamp) -> Optional[float]:
        """Value of a tag at a specific timestamp (nearest prior sample)."""
        if tag not in self.process_data.columns:
            return None
        df = self.process_data[["timestamp", tag]].dropna()
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        mask = df["timestamp"] <= timestamp
        if not mask.any():
            return None
        return float(df.loc[mask, tag].iloc[-1])

    def recent_mean(self, tag: str, minutes: int = 30) -> Optional[float]:
        """Average of a tag over the last N minutes of data."""
        if tag not in self.process_data.columns:
            return None
        df = self.process_data[["timestamp", tag]].dropna()
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].iloc[-1] - pd.Timedelta(minutes=minutes)
        recent = df.loc[df["timestamp"] >= cutoff, tag]
        if recent.empty:
            return None
        return float(recent.mean())

    def has_sessions_from(self, detectors: List[str],
                          tags: Optional[List[str]] = None) -> bool:
        """True if any co-occurring session comes from one of the given detectors
        (optionally also filtering by tag)."""
        for s in self.co_occurring_sessions:
            if s.detector not in detectors:
                continue
            if tags is None or s.tag in tags:
                return True
        return False


class DiagnosticRule(ABC):
    """
    A single diagnostic rule. Subclasses implement:
        - rule_name:   human-readable identifier
        - description: one-sentence intent
        - matches():   boolean, does this rule apply to this session?
        - fire():      produce a Recommendation when it matches.

    The engine iterates all rules for each session and collects the
    recommendations produced by all matching rules.
    """

    @property
    @abstractmethod
    def rule_name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def matches(self, session: EventSession, context: RuleContext) -> bool: ...

    @abstractmethod
    def fire(self, session: EventSession, context: RuleContext) -> Optional[Recommendation]: ...
