"""
Axion AI - LSTM Predictive Detector
====================================

Wraps an LSTMForecaster to emit alerts when the predicted future value of
a monitored variable crosses an operational limit. The output integrates
with Axion's alert/session/recommendation pipeline as a regular detector.

How it differs from TrendDetector
---------------------------------
TrendDetector fits a linear model to the recent window and projects forward.
That works for monotonic drifts but underestimates non-linear dynamics:
oscillations, S-curve degradations, transition slopes.

LSTMPredictiveDetector uses a learned multivariate model that captures:
  - Coupling between variables (cooling water flow → reactor temperature)
  - Non-linear settling after disturbances
  - Cyclic patterns (oscillation forecasting)
  - Lag structure (purity follows T_bot with a delay; the LSTM learns this)

Alert generation
----------------
For each evaluation:
  1. Take the most recent lookback window from the DataFrame
  2. Forecast all target variables across all configured horizons
  3. For each (target, horizon) cell:
       - Compare against the operational limit for that target
       - If predicted value violates the limit → emit alert with
         "time-to-violation" attribute = horizon
  4. Take only the EARLIEST predicted violation per (target, evaluation)
     to avoid emitting four redundant alerts at 5/15/30/60 min for the
     same future event.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from analytics.alerts import Alert, AlertType, Severity

from .lstm import LSTMForecaster


@dataclass
class PredictiveLimit:
    """
    Operational limit + violation logic for a single predicted variable.
    `direction` is "below" if going below `limit` is bad (e.g. purity_B),
    "above" if going above is bad (e.g. T_R_C).
    """
    tag: str
    limit: float
    direction: str             # "above" | "below"
    severity_at_limit: Severity = Severity.MEDIUM
    severity_far: Severity = Severity.HIGH
    far_offset: float = 1.0    # how far past `limit` triggers HIGH severity


# Default limits for the pilot's predicted KPIs
PILOT_PREDICTIVE_LIMITS = {
    "column.purity_B": PredictiveLimit(
        tag="column.purity_B", limit=98.5, direction="below",
        severity_at_limit=Severity.MEDIUM, severity_far=Severity.HIGH,
        far_offset=1.0,
    ),
    "cstr.T_R_C": PredictiveLimit(
        tag="cstr.T_R_C", limit=82.0, direction="above",
        severity_at_limit=Severity.MEDIUM, severity_far=Severity.HIGH,
        far_offset=2.0,
    ),
    # Q_reb is informational, not a violation per se, but we can still
    # track sustained excursions
    "column.Q_reb_kW": PredictiveLimit(
        tag="column.Q_reb_kW", limit=320.0, direction="above",
        severity_at_limit=Severity.LOW, severity_far=Severity.MEDIUM,
        far_offset=30.0,
    ),
}


class LSTMPredictiveDetector:
    """
    Forecast-based detector. Emits Alert objects when the LSTM forecaster
    predicts a future limit violation.
    """

    def __init__(
        self,
        forecaster: LSTMForecaster,
        limits: Optional[Dict[str, PredictiveLimit]] = None,
        cooldown_minutes: float = 30.0,
    ):
        self.forecaster = forecaster
        self.limits = limits or PILOT_PREDICTIVE_LIMITS
        self.cooldown_minutes = cooldown_minutes
        # Track the last-emitted-alert timestamp per (target, direction)
        # to avoid emitting the same forecast violation twice.
        self._last_emit: Dict[Tuple[str, str], pd.Timestamp] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """No additional fitting — the forecaster is pre-trained offline."""
        pass

    def run(self, df: pd.DataFrame) -> List[Alert]:
        """
        Walk the DataFrame in coarse strides (one evaluation per
        `eval_step_minutes`), call the forecaster, emit alerts for the
        earliest predicted violation per target.
        """
        if len(df) < self.forecaster.config.lookback_steps + 1:
            return []
        # Need feature columns
        missing = [f for f in self.forecaster.feature_cols if f not in df.columns]
        if missing:
            return []

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        W = self.forecaster.config.lookback_steps
        eval_step = max(1, int(round(15.0 / self.forecaster.config.sample_period_minutes)))
        # Evaluate every `eval_step` samples — full re-evaluation every
        # 15 minutes of process time (matches the typical operator
        # decision cadence).

        alerts: List[Alert] = []
        for end in range(W, len(df), eval_step):
            sub = df.iloc[: end + 1]
            now_ts = pd.Timestamp(sub["timestamp"].iloc[-1])
            forecast = self.forecaster.predict_from_df(sub)
            if forecast is None:
                continue

            for tgt, traj in forecast.items():
                if tgt not in self.limits:
                    continue
                lim = self.limits[tgt]
                violation_step = self._earliest_violation(traj, lim)
                if violation_step is None:
                    continue
                horizon_min = (violation_step + 1) * self.forecaster.config.sample_period_minutes
                # Cooldown
                key = (tgt, lim.direction)
                last = self._last_emit.get(key)
                if last is not None and (now_ts - last).total_seconds() < self.cooldown_minutes * 60:
                    continue

                predicted_value = float(traj[violation_step])
                current_value   = float(sub[tgt].iloc[-1])
                # Compute severity from how far past the limit the forecast goes
                if lim.direction == "below":
                    excess = lim.limit - predicted_value
                else:
                    excess = predicted_value - lim.limit
                severity = (lim.severity_far
                            if excess >= lim.far_offset
                            else lim.severity_at_limit)

                msg = (
                    f"LSTM forecasts {tgt} crossing limit ({lim.direction} "
                    f"{lim.limit:g}) in approximately {horizon_min:.0f} min "
                    f"(predicted {predicted_value:.2f}, current {current_value:.2f})."
                )
                alerts.append(Alert(
                    timestamp=now_ts,
                    detector="LSTM.Forecast",
                    alert_type=AlertType.TREND_PROJECTION,
                    severity=severity,
                    tag=tgt,
                    value=current_value,
                    limit=lim.limit,
                    confidence=0.75,
                    message=msg,
                    extra={
                        "predicted_value":     predicted_value,
                        "current_value":       current_value,
                        "horizon_minutes":     horizon_min,
                        "direction":           lim.direction,
                        "violation_excess":    excess,
                        "full_trajectory":     traj.tolist(),
                    },
                ))
                self._last_emit[key] = now_ts
        return alerts

    @staticmethod
    def _earliest_violation(
        trajectory: np.ndarray, lim: PredictiveLimit
    ) -> Optional[int]:
        """Return index of the earliest step where the limit is violated, or None."""
        if lim.direction == "below":
            mask = trajectory < lim.limit
        else:
            mask = trajectory > lim.limit
        if not mask.any():
            return None
        return int(np.argmax(mask))   # first True index
