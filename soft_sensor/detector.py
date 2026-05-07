"""
Axion AI - Soft Sensor Divergence Detector
==========================================

Compares a soft sensor's prediction against the actual measured value of
the same variable. A sustained disagreement between model and measurement
is operationally meaningful:

  - If the model is well-calibrated and the disagreement appears suddenly,
    the measurement is probably wrong (sensor drift, miscalibration, lab
    contamination).

  - If the disagreement develops gradually and correlates with a change in
    operating conditions, the process has drifted outside the model's
    training envelope — we're extrapolating, and the operator needs to
    know.

Either case is actionable. This detector emits alerts that a downstream
rule (R09_SoftSensorDivergence) translates into operator-facing
recommendations.

The detector is configured with:
  - A fitted SoftSensor
  - The name of the target variable (what the model predicts)
  - Absolute and relative tolerance bands
  - A window in which a sustained divergence triggers an alert

Logic
-----
For each sample, compute |actual - predicted|. If it exceeds
max(abs_tol, rel_tol * |actual|) for at least `min_duration_minutes`
consecutively, emit an alert. Severity scales with magnitude.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

from analytics.alerts import Alert, AlertType, Severity
from soft_sensor import SoftSensor


class SoftSensorDetector:
    """
    Detects sustained divergence between a soft sensor's prediction and
    the measured value of the same variable.
    """

    def __init__(
        self,
        sensor: SoftSensor,
        target_tag: str,
        abs_tolerance: float = 0.5,
        rel_tolerance: float = 0.01,
        min_duration_minutes: float = 10.0,
        cooldown_minutes: float = 30.0,
    ):
        self.sensor = sensor
        self.target_tag = target_tag
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
        self.min_duration_minutes = min_duration_minutes
        self.cooldown_minutes = cooldown_minutes

    def fit(self, df: pd.DataFrame) -> None:
        """No additional fitting — the underlying model is pre-trained."""
        pass

    def run(self, df: pd.DataFrame) -> List[Alert]:
        if self.target_tag not in df.columns:
            return []

        # Required feature columns must be present
        missing = [f for f in self.sensor.feature_names if f not in df.columns]
        if missing:
            return []

        alerts: List[Alert] = []
        X = df[self.sensor.feature_names]
        actual = df[self.target_tag].astype(float).values
        timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

        # Compute predictions
        try:
            preds, stds = self.sensor.predict_with_confidence(X)
        except Exception:
            return []

        # Absolute residual
        residuals = actual - preds

        # Adaptive threshold: max(abs_tol, rel_tol * |actual|, 3σ of the ensemble)
        abs_resid = np.abs(residuals)
        rel_thresh = self.rel_tolerance * np.abs(actual)
        model_thresh = 3.0 * stds    # trust ensemble uncertainty
        threshold = np.maximum.reduce([
            np.full_like(abs_resid, self.abs_tolerance),
            rel_thresh, model_thresh,
        ])
        diverging = abs_resid > threshold

        # Find sustained divergence windows
        if len(timestamps) < 2:
            return []
        dt_s = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
        if dt_s <= 0:
            dt_s = 60.0
        min_samples = max(3, int(self.min_duration_minutes * 60 / dt_s))
        cooldown_samples = int(self.cooldown_minutes * 60 / dt_s)

        last_alert = -cooldown_samples - 1
        i = 0
        n = len(diverging)
        while i < n:
            if not diverging[i]:
                i += 1
                continue
            # Walk forward to find the end of the divergence window
            j = i
            while j < n and diverging[j]:
                j += 1
            duration = j - i
            if duration >= min_samples and (i - last_alert) > cooldown_samples:
                # Peak of this window
                window_slice = slice(i, j)
                peak_idx = i + int(np.argmax(abs_resid[window_slice]))
                mag = float(abs_resid[peak_idx])

                # Severity from magnitude relative to threshold
                ratio = mag / max(threshold[peak_idx], 1e-6)
                if ratio > 4:
                    severity = Severity.HIGH
                elif ratio > 2:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                direction = "above" if residuals[peak_idx] > 0 else "below"
                alerts.append(Alert(
                    timestamp=timestamps.iloc[peak_idx],
                    detector="SoftSensor.Divergence",
                    alert_type=AlertType.TREND_PROJECTION,
                    severity=severity,
                    tag=self.target_tag,
                    value=float(actual[peak_idx]),
                    limit=float(preds[peak_idx]),
                    confidence=0.80,
                    message=(
                        f"{self.target_tag} measurement is {direction} soft sensor "
                        f"prediction by {mag:.3f} (threshold {threshold[peak_idx]:.3f}). "
                        f"Sustained for {duration * dt_s / 60:.0f} minutes."
                    ),
                    extra={
                        "actual":     float(actual[peak_idx]),
                        "predicted":  float(preds[peak_idx]),
                        "residual":   float(residuals[peak_idx]),
                        "threshold":  float(threshold[peak_idx]),
                        "duration_minutes": duration * dt_s / 60.0,
                        "ensemble_std": float(stds[peak_idx]),
                    },
                ))
                last_alert = i
            i = j

        return alerts
