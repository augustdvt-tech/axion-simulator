"""
Axion AI - LSTM Multi-Horizon Forecaster
=========================================

Predicts future values of multiple process variables across multiple
horizons simultaneously. Used to anticipate operational issues 5/15/30/60
minutes ahead and emit recommendations before specifications are violated.

Architecture
------------
A two-layer LSTM with dropout, followed by a Dense head that outputs
shape (max_horizon_steps, n_targets):

  Input:  (batch, lookback_steps, n_features)
            ↓
  LSTM(64, return_sequences=True) → Dropout(0.15)
            ↓
  LSTM(32) → Dropout(0.15)
            ↓
  Dense(max_horizon_steps × n_targets)   # flat
            ↓
  Reshape to (max_horizon_steps, n_targets)

Why a single multi-output head (instead of one model per (horizon, target)
combination):
  - Captures cross-target dependencies (T_R rising → purity falling)
  - Single forward pass at inference time (microseconds)
  - Smaller total parameter count than a zoo of models

Loss: MSE, equally weighted across all (horizon, target) cells. We could
weight nearer horizons more heavily — but in practice the equal-weight
version converges well and the operator gets uniform-quality predictions
across all horizons.

Inference returns either a single horizon prediction (e.g. forecast at
t+15min) or all horizons (the full trajectory). For UI we typically use
the latter for an overlay; for the detector we typically use specific
horizons.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import joblib

# Suppress most TF logging noise — we'll print our own progress
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras

from .windowing import (
    WindowConfig, Scaler,
    build_windows, build_windows_from_scenarios, time_split,
)


@dataclass
class LSTMMetrics:
    """Per-horizon, per-target metrics from training."""
    by_horizon: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history:   List[float] = field(default_factory=list)
    n_train: int = 0
    n_val:   int = 0

    def format(self) -> str:
        lines = [f"  n_train={self.n_train}  n_val={self.n_val}"]
        for h in sorted(self.by_horizon.keys()):
            lines.append(f"  horizon t+{h}min:")
            for tgt, m in self.by_horizon[h].items():
                lines.append(
                    f"    {tgt:25s}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
                    f"R²={m['r2']:.3f}"
                )
        return "\n".join(lines)


class LSTMForecaster:
    """
    Multi-horizon multi-target LSTM forecaster.

    Parameters
    ----------
    feature_cols : list of str
        Column names to use as model inputs.
    target_cols : list of str
        Column names to forecast. Each is predicted at every horizon.
    config : WindowConfig
        Lookback + forecast horizons + sample period.
    units1, units2 : int
        Hidden units in the two LSTM layers.
    dropout : float
        Dropout rate after each LSTM layer.
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_cols: List[str],
        config: WindowConfig,
        units1: int = 64,
        units2: int = 32,
        dropout: float = 0.15,
    ):
        self.feature_cols = list(feature_cols)
        self.target_cols  = list(target_cols)
        self.config       = config
        self.units1       = units1
        self.units2       = units2
        self.dropout      = dropout

        self.feature_scaler: Optional[Scaler] = None
        self.target_scaler:  Optional[Scaler] = None
        self.model:          Optional[keras.Model] = None
        self.metrics:        Optional[LSTMMetrics] = None

    # ---- model definition ----

    def _build_model(self) -> keras.Model:
        n_feat = len(self.feature_cols)
        n_tgt  = len(self.target_cols)
        H = self.config.max_horizon_steps

        inputs = keras.Input(shape=(self.config.lookback_steps, n_feat),
                             name="window")
        x = keras.layers.LSTM(self.units1, return_sequences=True,
                              name="lstm_1")(inputs)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.LSTM(self.units2, name="lstm_2")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        # Single dense head → (H * n_targets,) → reshape
        x = keras.layers.Dense(H * n_tgt, activation=None, name="head")(x)
        outputs = keras.layers.Reshape((H, n_tgt), name="forecast")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_forecaster")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # ---- training ----

    def fit(
        self,
        scenario_dfs: List[pd.DataFrame],
        epochs: int = 30,
        batch_size: int = 64,
        verbose: int = 1,
        val_fraction: float = 0.2,
    ) -> LSTMMetrics:
        """
        Train on multiple scenario DataFrames. Each is windowed independently
        (windows do not cross scenario boundaries). The combined dataset is
        time-split into train / val.
        """
        # Build all windows
        X, Y, ts = build_windows_from_scenarios(
            scenario_dfs, self.feature_cols, self.target_cols, self.config
        )
        if len(X) == 0:
            raise ValueError("No usable windows produced from scenarios")

        # Time-split BEFORE scaling (val data must not influence scaler)
        (X_tr, Y_tr), (X_val, Y_val) = time_split(X, Y, ts, val_fraction=val_fraction)

        # Fit scalers on training set only
        self.feature_scaler = Scaler().fit(X_tr)
        self.target_scaler  = Scaler().fit(Y_tr)

        X_tr_s  = self.feature_scaler.transform(X_tr)
        X_val_s = self.feature_scaler.transform(X_val)
        Y_tr_s  = self.target_scaler.transform(Y_tr)
        Y_val_s = self.target_scaler.transform(Y_val)

        # Build and train
        self.model = self._build_model()
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5,
            ),
        ]
        history = self.model.fit(
            X_tr_s, Y_tr_s,
            validation_data=(X_val_s, Y_val_s),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Compute per-horizon metrics on the validation set
        Y_val_pred_s = self.model.predict(X_val_s, batch_size=batch_size, verbose=0)
        Y_val_pred = self.target_scaler.inverse_transform(Y_val_pred_s)
        Y_val_true = Y_val   # already in original space

        metrics = LSTMMetrics(
            train_loss_history=list(history.history.get("loss", [])),
            val_loss_history=list(history.history.get("val_loss", [])),
            n_train=len(X_tr), n_val=len(X_val),
        )
        for h_min in self.config.horizons_minutes:
            h_step = int(round(h_min / self.config.sample_period_minutes)) - 1
            if h_step >= self.config.max_horizon_steps:
                continue
            metrics.by_horizon[h_min] = {}
            for j, tgt in enumerate(self.target_cols):
                pred = Y_val_pred[:, h_step, j]
                true = Y_val_true[:, h_step, j]
                resid = true - pred
                ss_res = float(np.sum(resid ** 2))
                ss_tot = float(np.sum((true - np.mean(true)) ** 2))
                metrics.by_horizon[h_min][tgt] = {
                    "mae":  float(np.mean(np.abs(resid))),
                    "rmse": float(np.sqrt(np.mean(resid ** 2))),
                    "r2":   1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
                }
        self.metrics = metrics
        return metrics

    # ---- inference ----

    def predict_window(self, X: np.ndarray) -> np.ndarray:
        """
        Predict from already-built windows.

        X: (N, lookback_steps, n_features)
        Returns: (N, max_horizon_steps, n_targets) in original units.
        """
        if self.model is None:
            raise RuntimeError("Forecaster not fitted")
        X_s = self.feature_scaler.transform(X.astype(np.float32))
        Y_s = self.model.predict(X_s, verbose=0)
        return self.target_scaler.inverse_transform(Y_s)

    def predict_from_df(self, df: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
        """
        Predict from a flat DataFrame. Builds the most recent lookback window
        and returns the forecast for ALL horizons.

        Returns a dict mapping target_col → array of shape (max_horizon_steps,)
        with predicted values in original units. Returns None if the DataFrame
        is too short to form a window.
        """
        if self.model is None:
            raise RuntimeError("Forecaster not fitted")
        W = self.config.lookback_steps
        if len(df) < W:
            return None
        sub = df[self.feature_cols].dropna().tail(W)
        if len(sub) < W:
            return None
        X = sub.to_numpy(dtype=np.float32)[None, :, :]   # (1, W, n_feat)
        Y_pred = self.predict_window(X)[0]               # (H, n_tgt)
        return {tgt: Y_pred[:, j] for j, tgt in enumerate(self.target_cols)}

    def predict_at_horizons(
        self, df: pd.DataFrame, horizons_minutes: Optional[List[int]] = None
    ) -> Optional[Dict[str, Dict[int, float]]]:
        """
        Convenience: return per-target, per-requested-horizon point predictions.

        Returns: { target_col: { horizon_min: predicted_value } }
        """
        full = self.predict_from_df(df)
        if full is None:
            return None
        horizons = horizons_minutes or self.config.horizons_minutes
        out: Dict[str, Dict[int, float]] = {}
        for tgt, traj in full.items():
            out[tgt] = {}
            for h_min in horizons:
                step = int(round(h_min / self.config.sample_period_minutes)) - 1
                if 0 <= step < len(traj):
                    out[tgt][h_min] = float(traj[step])
        return out

    # ---- persistence ----

    def save(self, path_dir: Path) -> None:
        """Save Keras weights + meta separately."""
        path_dir = Path(path_dir)
        path_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(path_dir / "model.keras")
        meta = {
            "feature_cols":   self.feature_cols,
            "target_cols":    self.target_cols,
            "config":         self.config,
            "units1":         self.units1,
            "units2":         self.units2,
            "dropout":        self.dropout,
            "feature_scaler": self.feature_scaler,
            "target_scaler":  self.target_scaler,
            "metrics":        self.metrics,
        }
        joblib.dump(meta, path_dir / "meta.joblib")

    @staticmethod
    def load(path_dir: Path) -> "LSTMForecaster":
        path_dir = Path(path_dir)
        meta = joblib.load(path_dir / "meta.joblib")
        forecaster = LSTMForecaster(
            feature_cols=meta["feature_cols"],
            target_cols=meta["target_cols"],
            config=meta["config"],
            units1=meta["units1"],
            units2=meta["units2"],
            dropout=meta["dropout"],
        )
        forecaster.feature_scaler = meta["feature_scaler"]
        forecaster.target_scaler  = meta["target_scaler"]
        forecaster.metrics        = meta["metrics"]
        forecaster.model          = keras.models.load_model(path_dir / "model.keras")
        return forecaster
