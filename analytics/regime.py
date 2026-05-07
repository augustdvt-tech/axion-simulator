"""
Axion AI - Regime Change Detector
=================================

Detects shifts between distinct operating regimes — e.g. when the process
transitions from steady state to a new operating point, from normal operation
to a degraded mode, or when a setpoint change pushes the process to a new
attractor.

Why this matters
----------------
SPC, PCA and trend detectors all answer: "is something wrong now?"
Regime change asks a different question: "did the process just *change mode*?"
This is essential because:

1. Many alerts that look anomalous are actually legitimate regime changes
   (a planned setpoint adjustment, a load change requested by production).
   Knowing "we just changed regime" lets the system avoid alerting on the
   normal transient that follows.

2. Some faults manifest precisely as unexplained regime changes — a degradation
   that pushes the system to a new equilibrium without any single variable
   being individually anomalous.

Method (CUSUM)
--------------
For each tag, run a two-sided CUSUM (Cumulative Sum) chart:
    S+_t = max(0, S+_{t-1} + (x_t - mu - k))
    S-_t = max(0, S-_{t-1} - (x_t - mu + k))
where mu is the baseline mean, k is the allowance (half the smallest shift we
care about, in sigma units), and an alarm is raised when S+ or S- exceeds h*sigma.

CUSUM was specifically designed for this problem and is far more sensitive than
Shewhart for sustained mean shifts. The classic parameters (k=0.5σ, h=5σ)
detect a 1σ shift with average run length ~10 samples, while raising essentially
zero false alarms during normal operation.

Engineering decisions for the MVP
---------------------------------
- Train baselines from initial 25% of the data (consistent with SPC)
- k = 0.5 sigma (detects ~1 sigma shifts)
- h = 5 (Page's recommendation for low FAR)
- After detection, the CUSUM is reset and a cooldown is enforced to avoid
  re-detecting the same regime change as multiple events.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .alerts import Alert, AlertType, Severity


@dataclass
class CUSUMBaseline:
    mean: float
    sigma: float


class RegimeChangeDetector:
    """
    Two-sided CUSUM detector for regime changes (sustained mean shifts).
    """

    def __init__(
        self,
        tags: List[str],
        k_sigma: float = 0.5,
        h_sigma: float = 5.0,
        cooldown_seconds: int = 1800,   # 30 min: regime changes are rare events
        training_fraction: float = 0.25,
    ):
        self.tags = list(tags)
        self.k_sigma = k_sigma
        self.h_sigma = h_sigma
        self.cooldown_seconds = cooldown_seconds
        self.training_fraction = training_fraction
        self.baselines: Dict[str, CUSUMBaseline] = {}

    # ------ fit ------
    def fit(self, df: pd.DataFrame) -> None:
        n_train = int(len(df) * self.training_fraction)
        train = df.iloc[:n_train]
        for tag in self.tags:
            if tag not in train.columns:
                continue
            series = train[tag].dropna()
            mu = float(series.mean())
            sig = float(series.std(ddof=1))
            if sig < 1e-9:
                sig = 1e-9
            self.baselines[tag] = CUSUMBaseline(mean=mu, sigma=sig)

    # ------ run ------
    def run(self, df: pd.DataFrame) -> List[Alert]:
        if not self.baselines:
            raise RuntimeError("RegimeChangeDetector must be fit() before run()")

        alerts: List[Alert] = []
        cooldown = pd.Timedelta(seconds=self.cooldown_seconds)
        timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

        for tag in self.tags:
            if tag not in df.columns or tag not in self.baselines:
                continue

            base = self.baselines[tag]
            k = self.k_sigma * base.sigma
            h = self.h_sigma * base.sigma
            values = df[tag].to_numpy(dtype=float)

            S_pos = 0.0
            S_neg = 0.0
            last_alert_ts: Optional[pd.Timestamp] = None

            for i, x in enumerate(values):
                if np.isnan(x):
                    continue
                deviation = x - base.mean
                S_pos = max(0.0, S_pos + deviation - k)
                S_neg = max(0.0, S_neg - deviation - k)

                triggered = None
                if S_pos > h:
                    triggered = ("upward", S_pos)
                elif S_neg > h:
                    triggered = ("downward", S_neg)

                if triggered is None:
                    continue
                ts = timestamps.iloc[i]
                if last_alert_ts is not None and ts - last_alert_ts <= cooldown:
                    # Reset and keep watching — but don't emit duplicate alert
                    S_pos = 0.0
                    S_neg = 0.0
                    continue

                direction, S_value = triggered
                # Estimate magnitude of the shift in sigma units
                # CUSUM crosses h after a shift of ~k+h/n, but a quick estimate:
                shift_estimate_sigma = float(deviation / base.sigma)
                severity = self._severity_from_shift(abs(shift_estimate_sigma))

                alerts.append(Alert(
                    timestamp=ts,
                    detector="Regime.CUSUM",
                    alert_type=AlertType.REGIME_CHANGE,
                    severity=severity,
                    tag=tag,
                    value=float(x),
                    limit=float(base.mean),
                    confidence=min(1.0, S_value / (2 * h)),
                    message=(
                        f"{tag}: regime change detected ({direction}). "
                        f"Current = {x:.3f}, baseline mean = {base.mean:.3f}, "
                        f"shift ≈ {shift_estimate_sigma:+.2f} sigma."
                    ),
                    extra={
                        "direction": direction,
                        "cusum_value": S_value,
                        "shift_sigma": shift_estimate_sigma,
                    },
                ))

                last_alert_ts = ts
                # Reset CUSUM after detection
                S_pos = 0.0
                S_neg = 0.0

        return alerts

    @staticmethod
    def _severity_from_shift(shift_sigma: float) -> Severity:
        if shift_sigma > 6:
            return Severity.CRITICAL
        if shift_sigma > 3:
            return Severity.HIGH
        if shift_sigma > 1.5:
            return Severity.MEDIUM
        return Severity.LOW
