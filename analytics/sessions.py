"""
Axion AI - Event Sessions
=========================

An Event Session groups a stream of related alerts into a single "event".
Without this, a sustained drift (say, a 24 h thermal drift) produces hundreds
of EWMA alerts that all describe the same underlying phenomenon. That is
overwhelming for the operator and drowns other, genuinely new events.

Grouping rule
-------------
Two alerts belong to the same session iff:
  - same detector (e.g. SPC.EWMA)
  - same tag (or both multivariate with None tag)
  - time gap between consecutive alerts < gap_minutes (default 30)

Output per session
------------------
- start_time / end_time
- peak_severity (max severity observed during the session)
- alert_count
- representative message (from the alert that triggered the session)
- peak_value, peak_deviation

This transforms a raw alert stream into a compact event log. This is exactly
what the Recommendation Engine (Task 4) will consume — and what operators
actually want to see on their screen.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd

from .alerts import Alert, Severity


_SEVERITY_ORDER = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


@dataclass
class EventSession:
    """A grouped sequence of related alerts forming a single event."""
    detector: str
    tag: Optional[str]
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_severity: Severity
    alert_count: int
    first_alert: Alert
    peak_alert: Alert
    contributors: Dict[str, float] = field(default_factory=dict)

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60.0

    def to_row(self) -> Dict[str, Any]:
        return {
            "detector":         self.detector,
            "tag":              self.tag,
            "start_time":       self.start_time,
            "end_time":         self.end_time,
            "duration_min":     round(self.duration_minutes, 2),
            "alert_count":      self.alert_count,
            "peak_severity":    self.peak_severity.value,
            "message":          self.peak_alert.message,
            "peak_value":       self.peak_alert.value,
        }


def group_alerts_into_sessions(
    alerts: List[Alert],
    gap_minutes: float = 30.0,
) -> List[EventSession]:
    """
    Group alerts by (detector, tag) and consecutive time proximity.

    Input alerts are assumed to come sorted by timestamp; if not, they are sorted.
    """
    if not alerts:
        return []

    alerts = sorted(alerts, key=lambda a: (a.detector, a.tag or "", a.timestamp))
    gap = pd.Timedelta(minutes=gap_minutes)

    sessions: List[EventSession] = []
    current: List[Alert] = []

    def flush():
        if not current:
            return
        sev = max((a.severity for a in current), key=lambda s: _SEVERITY_ORDER[s])
        peak = max(current, key=lambda a: (
            _SEVERITY_ORDER[a.severity],
            abs((a.value or 0) - (a.limit or 0)),
        ))
        # Merge contributors from all alerts in this session
        merged_contrib: Dict[str, float] = {}
        for a in current:
            for k, v in (a.contributors or {}).items():
                merged_contrib[k] = max(merged_contrib.get(k, 0.0), v)

        sessions.append(EventSession(
            detector=current[0].detector,
            tag=current[0].tag,
            start_time=current[0].timestamp,
            end_time=current[-1].timestamp,
            peak_severity=sev,
            alert_count=len(current),
            first_alert=current[0],
            peak_alert=peak,
            contributors=merged_contrib,
        ))

    for a in alerts:
        if not current:
            current = [a]
            continue
        prev = current[-1]
        same_stream = (a.detector == prev.detector and a.tag == prev.tag)
        close_in_time = (a.timestamp - prev.timestamp) <= gap
        if same_stream and close_in_time:
            current.append(a)
        else:
            flush()
            current = [a]
    flush()

    # Sort by start_time for easier consumption
    sessions.sort(key=lambda s: s.start_time)
    return sessions


def sessions_to_dataframe(sessions: List[EventSession]) -> pd.DataFrame:
    if not sessions:
        return pd.DataFrame(columns=[
            "detector", "tag", "start_time", "end_time", "duration_min",
            "alert_count", "peak_severity", "message", "peak_value",
        ])
    return pd.DataFrame([s.to_row() for s in sessions])
