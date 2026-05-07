"""
Axion AI - Frozen Sensor Detector
=================================

Detects frozen (stuck) sensors — a measurement that stops changing at all
for a period longer than what normal process variation and measurement
noise would allow.

Why a dedicated detector?
-------------------------
Classic SPC / PCA / CUSUM / trend detectors all assume the signal is alive
and compare its behavior against a baseline. A frozen sensor is the
opposite case: it stops being a signal at all. If the value it freezes
at is within 3σ of the baseline (the most common case — the sensor fails
while showing a reasonable number), NONE of the statistical detectors
will fire, even though the plant operator has effectively lost that
measurement.

Method
------
Sliding window: look at the standard deviation of the last N samples of
each tag. A real process variable with sensor noise has σ > 1e-4 (almost
always much more). If σ drops to exactly zero over a sustained window,
the sensor is frozen.

This is *not* just checking "all values equal" — that would fire on
setpoints and manipulated variables that legitimately stay at a fixed
value. We exclude those tags by providing an explicit list of
measured variables.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .alerts import Alert, AlertType, Severity


# Tags we consider to be real sensor measurements. For setpoints /
# manipulated variables, zero variance is normal behavior.
DEFAULT_MEASURED_TAGS = [
    "cstr.T_R_C", "cstr.T_J_C", "cstr.C_A", "cstr.conversion",
    "column.x_D", "column.x_B_A", "column.purity_B",
    "column.T_top_C", "column.T_bot_C", "column.Q_reb_kW",
]


class FrozenSensorDetector:
    """
    Looks for sensors stuck at a constant value over a sustained window.

    Parameters
    ----------
    tags : list of tag names to monitor (measured variables only)
    window_minutes : size of the sliding window (default: 20)
    min_std : minimum acceptable standard deviation (default: 1e-4).
              Any σ below this over the window is considered a fault.
    cooldown_seconds : dedup window (default: 1800s = 30 min)
    """

    def __init__(
        self,
        tags: Optional[List[str]] = None,
        window_minutes: int = 20,
        min_std: float = 1e-4,
        cooldown_seconds: int = 1800,
    ):
        self.tags = tags if tags is not None else list(DEFAULT_MEASURED_TAGS)
        self.window_minutes = window_minutes
        self.min_std = min_std
        self.cooldown_seconds = cooldown_seconds

    def fit(self, df: pd.DataFrame) -> None:
        """No fitting needed — detector is parameter-free beyond config."""
        pass

    def run(self, df: pd.DataFrame) -> List[Alert]:
        alerts: List[Alert] = []
        cooldown = pd.Timedelta(seconds=self.cooldown_seconds)
        timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

        if len(timestamps) < 2:
            return alerts

        # Derive sampling period from first two timestamps
        dt_s = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
        if dt_s <= 0:
            dt_s = 60.0
        window_size = max(5, int(self.window_minutes * 60 / dt_s))

        for tag in self.tags:
            if tag not in df.columns:
                continue

            values = df[tag].to_numpy(dtype=float)
            last_alert_ts: Optional[pd.Timestamp] = None

            # Slide window; check σ of each window
            for end in range(window_size, len(values)):
                window = values[end - window_size: end]
                if np.any(np.isnan(window)):
                    continue
                w_std = float(np.std(window, ddof=0))
                if w_std >= self.min_std:
                    continue

                ts = timestamps.iloc[end]
                if last_alert_ts is not None and ts - last_alert_ts <= cooldown:
                    continue

                stuck_value = float(window[-1])
                alerts.append(Alert(
                    timestamp=ts,
                    detector="FrozenSensor",
                    alert_type=AlertType.SHEWHART_VIOLATION,  # reuse type
                    severity=Severity.HIGH,
                    tag=tag,
                    value=stuck_value,
                    limit=self.min_std,
                    confidence=0.95,
                    message=(
                        f"{tag} has been stuck at {stuck_value:.4f} for the "
                        f"last {self.window_minutes} minutes (σ < "
                        f"{self.min_std:.0e}). Sensor may be frozen."
                    ),
                    extra={"window_std": w_std, "window_minutes": self.window_minutes},
                ))
                last_alert_ts = ts

        return alerts
