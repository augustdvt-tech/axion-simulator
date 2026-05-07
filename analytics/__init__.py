"""
Axion AI - Analytical Engine
============================

The analytical core of Axion AI. Detects deviations, anomalies and changes in
process behavior from time-series data.

Detectors:
- SPCDetector:           Univariate Shewhart + EWMA control charts
- PCADetector:           Multivariate Hotelling T2 + SPE/Q
- TrendDetector:         Linear projection with time-to-limit estimation
- RegimeChangeDetector:  CUSUM for sustained mean shifts

The AnalyticalEngine orchestrates all four and returns either a raw alerts
timeline or a grouped EventSession timeline.
"""

from .alerts import Alert, AlertType, Severity, alerts_to_dataframe
from .sessions import EventSession, group_alerts_into_sessions, sessions_to_dataframe
from .spc import SPCDetector, TagBaseline
from .pca import PCADetector, PCAModel
from .trend import TrendDetector, PILOT_OPERATIONAL_LIMITS
from .regime import RegimeChangeDetector
from .frozen import FrozenSensorDetector, DEFAULT_MEASURED_TAGS
from .engine import AnalyticalEngine, PILOT_TAGS

__all__ = [
    "Alert", "AlertType", "Severity", "alerts_to_dataframe",
    "EventSession", "group_alerts_into_sessions", "sessions_to_dataframe",
    "SPCDetector", "TagBaseline",
    "PCADetector", "PCAModel",
    "TrendDetector", "PILOT_OPERATIONAL_LIMITS",
    "RegimeChangeDetector",
    "FrozenSensorDetector", "DEFAULT_MEASURED_TAGS",
    "AnalyticalEngine", "PILOT_TAGS",
]
