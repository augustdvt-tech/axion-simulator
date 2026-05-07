"""
Axion AI - Predictive (LSTM Forecasting) Module
================================================

Multi-horizon, multi-target LSTM forecasting of process variables.
Anticipates future operating excursions before they happen, emitting
alerts and recommendations with time-to-violation estimates.

Components:
  - WindowConfig:           lookback + horizons + sample period config
  - Scaler:                 z-normalization helper
  - LSTMForecaster:         the model (TF/Keras-based)
  - LSTMMetrics:            per-horizon training diagnostics
  - LSTMPredictiveDetector: emits alerts when forecasts predict limit violations
  - PredictiveLimit:        operational limit + violation logic
  - PILOT_PREDICTIVE_LIMITS: default limits for the pilot
"""

from .windowing import (
    WindowConfig, Scaler,
    build_windows, build_windows_from_scenarios, time_split,
)
try:
    from .lstm import LSTMForecaster, LSTMMetrics
    from .detector import (
        LSTMPredictiveDetector, PredictiveLimit, PILOT_PREDICTIVE_LIMITS,
    )
except ImportError:
    pass

__all__ = [
    "WindowConfig", "Scaler",
    "build_windows", "build_windows_from_scenarios", "time_split",
    "LSTMForecaster", "LSTMMetrics",
    "LSTMPredictiveDetector", "PredictiveLimit", "PILOT_PREDICTIVE_LIMITS",
]
