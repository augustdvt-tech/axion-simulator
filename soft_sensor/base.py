"""
Axion AI - Soft Sensor Base
===========================

A soft sensor estimates a variable that is hard to measure directly (slow,
expensive, or intermittent) from variables that are easy to measure (fast,
continuous, cheap). The classic example in distillation is estimating
product purity — which normally requires a gas chromatograph sampling every
15-30 minutes — from tray temperatures and flow rates that are measured
every second.

Why this matters operationally
------------------------------
Without a soft sensor, operators either:
  (a) wait for the next GC sample (losing 15-30 min of control authority
      when the process drifts), or
  (b) use a rough manual correlation (T_bot -> purity) that works in a
      narrow band but fails in transitions.

With a calibrated soft sensor, purity is estimated continuously. This lets:
  - Rule R06_PurityDeviation fire on a continuous signal instead of a
    stepped one — the engine can detect drift faster
  - The operator see current (predicted) purity at all times
  - The control loop close on purity directly (future: MPC)

Design
------
This module provides:
  - SoftSensor: abstract base — fit(), predict(), predict_with_confidence()
  - PuritySoftSensor: concrete random-forest implementation for the pilot
  - EnsembleSoftSensor: combines multiple models for uncertainty estimation
  - save/load: joblib-based persistence for production deployment

The interface is deliberately simple. A soft sensor's job is to take a
DataFrame slice of secondary variables and return a predicted value — the
rest of Axion AI does not care whether the underlying model is a random
forest, XGBoost, a neural network, or a hand-built linear correlation.

Training data
-------------
For the simulator-driven MVP, training uses the CSV scenarios because they
contain both the secondary variables (T_bot, Q_reb, etc.) AND the ground
truth purity from the simulator. In production, training would combine:
  - Historical DCS data (the secondary variables)
  - Historical LIMS records (the ground truth purity from the GC/lab)

Once enough labeled pairs exist (>500 is a reasonable starting point), the
soft sensor trains offline and is then applied online continuously.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib


@dataclass
class SoftSensorMetrics:
    """Diagnostic metrics from a soft sensor training run."""
    mae: float                      # Mean absolute error on test set
    rmse: float                     # Root mean squared error
    r2: float                       # Coefficient of determination
    bias: float                     # Mean signed residual (actual - predicted)
    max_error: float                # Worst-case absolute error
    n_train: int
    n_test: int
    feature_names: List[str]
    target_name: str

    def format(self) -> str:
        return (
            f"MAE={self.mae:.3f}  RMSE={self.rmse:.3f}  R²={self.r2:.3f}  "
            f"bias={self.bias:+.3f}  max_err={self.max_error:.3f}  "
            f"n_train={self.n_train}  n_test={self.n_test}"
        )


class SoftSensor(ABC):
    """
    Abstract base for any soft sensor. Subclasses implement the underlying
    ML model but share the same interface.
    """

    # Set by subclass or by fit()
    feature_names: List[str]
    target_name: str
    metrics: Optional[SoftSensorMetrics]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> SoftSensorMetrics: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_with_confidence(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (prediction, std) where std is per-sample uncertainty.

        Default implementation returns zero uncertainty. Subclasses like
        EnsembleSoftSensor override this.
        """
        preds = self.predict(X)
        return preds, np.zeros_like(preds)

    # ---- convenience helpers ----

    def predict_from_row(self, row: dict) -> float:
        """Convenience for single-sample prediction from a dict of tags."""
        X = pd.DataFrame([{f: row.get(f, np.nan) for f in self.feature_names}])
        return float(self.predict(X)[0])

    def save(self, path: Path) -> None:
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: Path) -> "SoftSensor":
        return joblib.load(Path(path))


# =============================================================================
# Utility: compute metrics from predictions vs actuals
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    target_name: str,
    n_train: int,
    n_test: int,
) -> SoftSensorMetrics:
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return SoftSensorMetrics(
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals ** 2))),
        r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        bias=float(np.mean(residuals)),
        max_error=float(np.max(np.abs(residuals))),
        n_train=n_train,
        n_test=n_test,
        feature_names=list(feature_names),
        target_name=target_name,
    )
