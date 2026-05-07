"""
Axion AI - Soft Sensor Module
=============================

ML-based estimation of variables that are difficult to measure directly.
The primary use case is product purity: normally measured by slow or
intermittent lab analysis (GC), estimated continuously from fast secondary
variables (temperatures, flows, reflux).

Components:
  - SoftSensor:        abstract base class
  - PuritySoftSensor:  gradient-boosted ensemble for pilot process purity
  - SoftSensorMetrics: training diagnostics dataclass
"""

from .base import SoftSensor, SoftSensorMetrics, compute_metrics
from .purity import PuritySoftSensor, PILOT_PURITY_FEATURES, TARGET_PURITY
from .detector import SoftSensorDetector

__all__ = [
    "SoftSensor", "SoftSensorMetrics", "compute_metrics",
    "PuritySoftSensor", "PILOT_PURITY_FEATURES", "TARGET_PURITY",
    "SoftSensorDetector",
]
