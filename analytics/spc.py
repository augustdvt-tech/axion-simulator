"""
Axion AI - SPC Detector (Statistical Process Control)
=====================================================

Univariate Statistical Process Control. Two classic techniques:

1. SHEWHART chart: detects large, sudden deviations (one point outside +/- 3 sigma).
   Fast and robust. Misses small persistent shifts.

2. EWMA chart (Exponentially Weighted Moving Average): detects small, persistent
   drifts and shifts that Shewhart misses. Slower to react but more sensitive.

Both charts are computed for every monitored tag. The control limits (mean,
sigma) are estimated from a "training window" of normal operation, and then
fixed during monitoring (this is how it's done in real industrial SPC — limits
should not auto-adjust to drifts, that's exactly what we want to detect).

Engineering decisions for the MVP:
- Training window: first 25% of the data (configurable)
- Shewhart: K = 3 sigma (standard)
- EWMA: lambda = 0.2, control limit at L = 3
- Each detector emits one Alert per violation and then enters a cooldown
  (default 10 min) to avoid alert spam from a single sustained event.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .alerts import Alert, AlertType, Severity


# =============================================================================
# Per-tag SPC state
# =============================================================================

@dataclass
class TagBaseline:
    """Mean, sigma and EWMA control limits for a single tag, fitted from data."""
    mean: float
    sigma: float
    ewma_lambda: float
    ewma_L: float

    @property
    def shewhart_ucl(self) -> float:
        return self.mean + 3 * self.sigma

    @property
    def shewhart_lcl(self) -> float:
        return self.mean - 3 * self.sigma

    @property
    def ewma_steady_sigma(self) -> float:
        """Asymptotic sigma of the EWMA statistic at steady state."""
        lam = self.ewma_lambda
        return self.sigma * np.sqrt(lam / (2 - lam))

    @property
    def ewma_ucl(self) -> float:
        return self.mean + self.ewma_L * self.ewma_steady_sigma

    @property
    def ewma_lcl(self) -> float:
        return self.mean - self.ewma_L * self.ewma_steady_sigma


# =============================================================================
# SPC Detector
# =============================================================================

class SPCDetector:
    """
    Univariate Shewhart + EWMA detector for a set of tags.

    Workflow:
        detector = SPCDetector(tags=["cstr.T_R_C", "column.purity_B"])
        detector.fit(df_training)         # learn baselines from normal data
        alerts = detector.run(df_full)    # emit alerts on the full dataset
    """

    def __init__(
        self,
        tags: List[str],
        ewma_lambda: float = 0.2,
        ewma_L: float = 3.0,
        cooldown_seconds: int = 600,
        training_fraction: float = 0.25,
    ):
        self.tags = list(tags)
        self.ewma_lambda = ewma_lambda
        self.ewma_L = ewma_L
        self.cooldown_seconds = cooldown_seconds
        self.training_fraction = training_fraction
        self.baselines: Dict[str, TagBaseline] = {}

    # ------ fit ------
    def fit(self, df: pd.DataFrame) -> None:
        """Fit baselines (mean, sigma) using a training slice."""
        n_train = int(len(df) * self.training_fraction)
        train = df.iloc[:n_train]
        for tag in self.tags:
            if tag not in train.columns:
                continue
            series = train[tag].dropna()
            mu = float(series.mean())
            sig = float(series.std(ddof=1))
            # Avoid degenerate baselines (constant signal)
            if sig < 1e-9:
                sig = 1e-9
            self.baselines[tag] = TagBaseline(
                mean=mu, sigma=sig,
                ewma_lambda=self.ewma_lambda, ewma_L=self.ewma_L,
            )

    # ------ run ------
    def run(self, df: pd.DataFrame) -> List[Alert]:
        """
        Run the detector over the entire DataFrame.

        Returns one alert per violation event (cooldown deduplicated).
        """
        if not self.baselines:
            raise RuntimeError("SPCDetector must be fit() before run()")

        alerts: List[Alert] = []
        cooldown = pd.Timedelta(seconds=self.cooldown_seconds)

        timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True)

        for tag in self.tags:
            if tag not in df.columns or tag not in self.baselines:
                continue

            base = self.baselines[tag]
            values = df[tag].to_numpy(dtype=float)

            # State vars for this tag
            ewma = base.mean
            last_shewhart_alert: Optional[pd.Timestamp] = None
            last_ewma_alert: Optional[pd.Timestamp] = None

            for i, x in enumerate(values):
                if np.isnan(x):
                    continue
                ts = timestamps.iloc[i]

                # --- Shewhart check ---
                if x > base.shewhart_ucl or x < base.shewhart_lcl:
                    if last_shewhart_alert is None or ts - last_shewhart_alert > cooldown:
                        side = "above" if x > base.shewhart_ucl else "below"
                        limit = base.shewhart_ucl if x > base.shewhart_ucl else base.shewhart_lcl
                        deviation = abs(x - base.mean) / base.sigma
                        alerts.append(Alert(
                            timestamp=ts,
                            detector="SPC.Shewhart",
                            alert_type=AlertType.SHEWHART_VIOLATION,
                            severity=self._severity_from_sigma(deviation),
                            tag=tag,
                            value=float(x),
                            limit=float(limit),
                            confidence=min(1.0, deviation / 3.0),
                            message=(
                                f"{tag} = {x:.3f} is {side} 3-sigma "
                                f"(limit {limit:.3f}, baseline {base.mean:.3f} ± {base.sigma:.3f})"
                            ),
                            extra={"deviation_sigma": deviation},
                        ))
                        last_shewhart_alert = ts

                # --- EWMA update + check ---
                ewma = self.ewma_lambda * x + (1 - self.ewma_lambda) * ewma
                if ewma > base.ewma_ucl or ewma < base.ewma_lcl:
                    if last_ewma_alert is None or ts - last_ewma_alert > cooldown:
                        side = "above" if ewma > base.ewma_ucl else "below"
                        limit = base.ewma_ucl if ewma > base.ewma_ucl else base.ewma_lcl
                        deviation = abs(ewma - base.mean) / base.ewma_steady_sigma
                        alerts.append(Alert(
                            timestamp=ts,
                            detector="SPC.EWMA",
                            alert_type=AlertType.EWMA_VIOLATION,
                            severity=self._severity_from_sigma(deviation),
                            tag=tag,
                            value=float(ewma),
                            limit=float(limit),
                            confidence=min(1.0, deviation / 3.0),
                            message=(
                                f"{tag}: EWMA = {ewma:.3f} is {side} control "
                                f"limit ({limit:.3f}). Sustained drift detected."
                            ),
                            extra={"deviation_sigma": deviation, "ewma_lambda": self.ewma_lambda},
                        ))
                        last_ewma_alert = ts

        return alerts

    @staticmethod
    def _severity_from_sigma(dev: float) -> Severity:
        """Map deviation magnitude to severity bucket."""
        if dev > 8:
            return Severity.CRITICAL
        if dev > 5:
            return Severity.HIGH
        if dev > 3.5:
            return Severity.MEDIUM
        return Severity.LOW
