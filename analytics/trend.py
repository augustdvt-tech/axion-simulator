"""
Axion AI - Trend Detector with Time-to-Limit Projection
=======================================================

Linear projection of a slow-moving variable to estimate "how many minutes until
this variable exceeds its operational limit". This is the first proactive
component of Axion AI: instead of waiting for the limit to be crossed, we
extrapolate the current trend and warn the operator in advance.

Why this matters
----------------
SPC tells you the variable is *out of control*; trend projection tells you
*when it will be out of spec*. These are different operational events. A drift
that takes 10 hours to cross the limit gives the engineer plenty of time to
act if she knows about it; without projection, she only learns about it when
it's too late.

Method
------
- For each tag, fit a linear regression on a sliding window of the latest N
  samples (default: 30 minutes).
- Use the slope to project where the variable will be in T minutes ahead.
- If the projection crosses an operational limit AND the slope is statistically
  significant (R^2 above threshold and slope above noise threshold), emit an
  alert with the estimated "time to limit".

Alerts include:
- Current value, projected value, slope (per minute)
- Estimated minutes until limit is crossed
- Confidence based on R^2 of the fit

Engineering decisions for the MVP
---------------------------------
- Window: 30 minutes (configurable). Smaller window = more responsive but noisier.
- Min R^2 to consider trend significant: 0.5
- Operational limits per tag come from the simulator's design specifications
  (passed in by the caller — these are *operational* limits, not the *control*
  limits learned by SPC).
- Cooldown deduplication, same as other detectors.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .alerts import Alert, AlertType, Severity


# =============================================================================
# Operational limits (default for the pilot process)
# =============================================================================

# These are *operational* limits (what makes the operator unhappy), not the
# *control* limits (statistical 3-sigma). Equivalent to the engineering limits
# defined in the architecture document, section 3.2.
PILOT_OPERATIONAL_LIMITS = {
    "cstr.T_R_C":      {"low": 75.0,  "high": 83.0},
    "cstr.T_J_C":      {"low": 50.0,  "high": 65.0},
    "cstr.conversion": {"low": 0.78,  "high": None},
    "cstr.C_A":        {"low": None,  "high": 250.0},
    "column.purity_B": {"low": 98.5,  "high": None},
    "column.T_top_C":  {"low": 65.0,  "high": 78.0},
    "column.T_bot_C":  {"low": 110.0, "high": 122.0},
    "column.Q_reb_kW": {"low": None,  "high": 320.0},
}


# =============================================================================
# Trend detector
# =============================================================================

class TrendDetector:
    """
    Linear-projection trend detector with time-to-limit estimation.

        detector = TrendDetector(
            tags=["cstr.T_R_C", "column.purity_B"],
            limits=PILOT_OPERATIONAL_LIMITS,
        )
        alerts = detector.run(df)
    """

    def __init__(
        self,
        tags: List[str],
        limits: Dict[str, Dict[str, Optional[float]]],
        window_minutes: int = 60,
        horizon_minutes: int = 240,
        min_r_squared: float = 0.25,
        smoothing_minutes: int = 5,
        cooldown_seconds: int = 1800,
    ):
        self.tags = list(tags)
        self.limits = limits
        self.window_minutes = window_minutes
        self.horizon_minutes = horizon_minutes
        self.min_r_squared = min_r_squared
        self.smoothing_minutes = smoothing_minutes
        self.cooldown_seconds = cooldown_seconds

    # ------ helper: linear fit ------
    @staticmethod
    def _linear_fit(t_minutes: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (slope_per_minute, intercept, r_squared).
        Robust to constant signals (returns 0 slope, R²=0).
        """
        n = len(t_minutes)
        if n < 3:
            return 0.0, float(np.mean(y)), 0.0
        t_mean = t_minutes.mean()
        y_mean = y.mean()
        denom = np.sum((t_minutes - t_mean) ** 2)
        if denom < 1e-12:
            return 0.0, float(y_mean), 0.0
        slope = float(np.sum((t_minutes - t_mean) * (y - y_mean)) / denom)
        intercept = float(y_mean - slope * t_mean)
        # R²
        ss_tot = float(np.sum((y - y_mean) ** 2))
        if ss_tot < 1e-12:
            return slope, intercept, 0.0
        y_pred = slope * t_minutes + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        return slope, intercept, r2

    # ------ helper: time to limit ------
    @staticmethod
    def _time_to_limit(current: float, slope_per_min: float,
                       low: Optional[float], high: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns (minutes_to_limit, breach_value) given current value and slope.
        If slope is zero or moving away from any limit, returns (None, None).
        """
        if abs(slope_per_min) < 1e-9:
            return None, None
        # Moving up — check upper limit
        if slope_per_min > 0 and high is not None and current < high:
            return (high - current) / slope_per_min, high
        # Moving down — check lower limit
        if slope_per_min < 0 and low is not None and current > low:
            return (low - current) / slope_per_min, low
        return None, None

    # ------ run ------
    def run(self, df: pd.DataFrame) -> List[Alert]:
        alerts: List[Alert] = []
        cooldown = pd.Timedelta(seconds=self.cooldown_seconds)

        timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
        t_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()

        # Determine window size in samples (assume uniform sampling rate)
        if len(t_seconds) > 1:
            dt_s = t_seconds[1] - t_seconds[0]
        else:
            dt_s = 60.0
        window_size = max(5, int(self.window_minutes * 60 / dt_s))
        # Step every minute (don't fit at every sample — too expensive and noisy)
        step = max(1, int(60 / dt_s))

        for tag in self.tags:
            if tag not in df.columns or tag not in self.limits:
                continue
            lim = self.limits[tag]
            low, high = lim.get("low"), lim.get("high")
            if low is None and high is None:
                continue

            values = df[tag].to_numpy(dtype=float)

            # Smooth with a rolling mean to reduce measurement noise. This is
            # critical: linear regression on noisy raw data has very low R²
            # even when a real trend is present.
            if self.smoothing_minutes > 0:
                smooth_size = max(2, int(self.smoothing_minutes * 60 / dt_s))
                values = pd.Series(values).rolling(
                    window=smooth_size, min_periods=1, center=False
                ).mean().to_numpy()

            last_alert_ts: Optional[pd.Timestamp] = None

            for end in range(window_size, len(values), step):
                start = end - window_size
                t_window = t_seconds[start:end] / 60.0     # convert to minutes
                y_window = values[start:end]
                if np.any(np.isnan(y_window)):
                    continue

                slope, intercept, r2 = self._linear_fit(t_window, y_window)
                if r2 < self.min_r_squared:
                    continue

                current = float(y_window[-1])
                t_to_lim, breach_val = self._time_to_limit(current, slope, low, high)
                if t_to_lim is None or t_to_lim < 0 or t_to_lim > self.horizon_minutes:
                    continue

                ts = timestamps.iloc[end - 1]
                if last_alert_ts is not None and ts - last_alert_ts <= cooldown:
                    continue

                projected = current + slope * t_to_lim
                severity = self._severity_from_time(t_to_lim)

                alerts.append(Alert(
                    timestamp=ts,
                    detector="Trend.Projection",
                    alert_type=AlertType.TREND_PROJECTION,
                    severity=severity,
                    tag=tag,
                    value=current,
                    limit=float(breach_val),
                    confidence=float(r2),
                    message=(
                        f"{tag} = {current:.3f}, trending {slope:+.4f}/min "
                        f"(R²={r2:.2f}). Projected to reach limit "
                        f"{breach_val:.3f} in ~{t_to_lim:.1f} minutes."
                    ),
                    extra={
                        "slope_per_minute": slope,
                        "minutes_to_limit": t_to_lim,
                        "r_squared": r2,
                        "projected_value": projected,
                    },
                ))
                last_alert_ts = ts

        return alerts

    @staticmethod
    def _severity_from_time(minutes_to_limit: float) -> Severity:
        """Closer to the limit = higher urgency."""
        if minutes_to_limit < 5:
            return Severity.CRITICAL
        if minutes_to_limit < 15:
            return Severity.HIGH
        if minutes_to_limit < 30:
            return Severity.MEDIUM
        return Severity.LOW
