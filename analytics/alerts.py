"""
Axion AI - Alert System
=======================

Common data structures for alerts emitted by every analytical detector.

Every detector (SPC, PCA, trend, regime) emits Alert objects with a uniform
shape. This is what the future Recommendation Engine (Task 4) will consume to
diagnose what's happening and propose what to do about it.

Design choice: an Alert is intentionally lightweight (no methods) because
serialization to JSON / DB rows / message queue payloads must be trivial.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    # Univariate SPC
    SHEWHART_VIOLATION = "shewhart_violation"   # outside 3-sigma
    EWMA_VIOLATION = "ewma_violation"           # smoothed control limit
    NELSON_RULE = "nelson_rule"                 # pattern in last N samples

    # Multivariate
    HOTELLING_T2 = "hotelling_t2"               # T2 above control limit
    SPE_Q = "spe_q"                             # SPE/Q above control limit

    # Trend / forecast
    TREND_PROJECTION = "trend_projection"       # variable will exceed limit in T minutes

    # Regime
    REGIME_CHANGE = "regime_change"             # operating regime change detected


@dataclass
class Alert:
    """A single alert emitted by a detector."""

    timestamp: pd.Timestamp                     # when the alert occurred
    detector: str                               # which detector emitted it
    alert_type: AlertType                       # what kind of alert
    severity: Severity
    tag: Optional[str] = None                   # affected variable (None for multivariate)
    message: str = ""                           # human-readable description
    value: Optional[float] = None               # current value of the offending variable
    limit: Optional[float] = None               # the limit that was violated
    confidence: float = 1.0                     # 0..1 confidence
    contributors: Dict[str, float] = field(default_factory=dict)  # variable -> contribution
    extra: Dict[str, Any] = field(default_factory=dict)            # detector-specific info

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alert_type"] = self.alert_type.value
        d["severity"] = self.severity.value
        d["timestamp"] = self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp)
        return d


def alerts_to_dataframe(alerts) -> pd.DataFrame:
    """Convert a list of Alert objects to a DataFrame for analysis/plotting."""
    if not alerts:
        return pd.DataFrame(columns=[
            "timestamp", "detector", "alert_type", "severity", "tag",
            "message", "value", "limit", "confidence",
        ])
    rows = []
    for a in alerts:
        rows.append({
            "timestamp":   a.timestamp,
            "detector":    a.detector,
            "alert_type":  a.alert_type.value,
            "severity":    a.severity.value,
            "tag":         a.tag,
            "message":     a.message,
            "value":       a.value,
            "limit":       a.limit,
            "confidence":  a.confidence,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)
