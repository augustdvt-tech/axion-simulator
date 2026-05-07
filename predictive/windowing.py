"""
Axion AI - Windowed Dataset Utility
====================================

LSTM forecasting requires data in the form (window of past, future targets).
This module provides utilities to build such windows from a flat time-series
DataFrame.

Conceptual layout
-----------------
For each timestamp t, we extract:
  - X[t] = [x_{t-W+1}, ..., x_{t}]   shape (W, n_features)
  - Y[t] = [y_{t+1},  ..., y_{t+H}]  shape (H, n_targets)
  where W is the lookback window and H is the forecast horizon (max).

The sequences slide forward in time with stride 1, producing N samples per
DataFrame of length N+W+H-1.

For training, we further split into train/val sets by time (NOT random) —
random splitting leaks future into past. The split point is configurable;
the default reserves the last 20% of timestamps for validation.

Standardization
---------------
Inputs are z-normalized using mean/std of the TRAINING set only. The same
scaler is applied at inference time. Targets are z-normalized too: the
LSTM predicts in normalized space, and we de-normalize at inference for
human-readable values.

Multi-target structure
----------------------
We support predicting multiple variables simultaneously (e.g. purity, T_R,
Q_reb at the same time). The model output shape is (H, n_targets). Each
target gets its own normalization parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class WindowConfig:
    """Configuration for building forecast windows."""
    lookback_minutes: int = 60          # past samples used as context
    horizons_minutes: List[int] = field(default_factory=lambda: [5, 15, 30, 60])
    sample_period_minutes: float = 1.0  # CSV sample period

    @property
    def lookback_steps(self) -> int:
        return int(round(self.lookback_minutes / self.sample_period_minutes))

    @property
    def horizon_steps(self) -> List[int]:
        """Forecast horizons converted to step counts."""
        return [int(round(h / self.sample_period_minutes)) for h in self.horizons_minutes]

    @property
    def max_horizon_steps(self) -> int:
        return max(self.horizon_steps)


@dataclass
class Scaler:
    """Per-column z-normalization. Fit on train, apply everywhere."""
    means: np.ndarray = field(default_factory=lambda: np.array([]))
    stds:  np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, X: np.ndarray) -> "Scaler":
        # X shape (N, F) — flatten across time/sequence for stats
        flat = X.reshape(-1, X.shape[-1])
        self.means = flat.mean(axis=0)
        self.stds  = flat.std(axis=0)
        # Avoid division by zero on constant columns
        self.stds[self.stds < 1e-9] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.means) / self.stds

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.stds + self.means


def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window arrays from a single DataFrame.

    Returns
    -------
    X : array shape (N, lookback_steps, n_features)
        Past windows of feature values.
    Y : array shape (N, max_horizon_steps, n_targets)
        Future windows of target values for ALL horizons up to max.
    timestamps : array shape (N,)
        The timestamp corresponding to the *current* sample (the last in
        each X window). Useful for evaluation indexing.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must include a 'timestamp' column")
    needed = list(set(feature_cols) | set(target_cols))
    df = df[["timestamp"] + needed].dropna().reset_index(drop=True)
    if len(df) == 0:
        return np.empty((0, config.lookback_steps, len(feature_cols))), \
               np.empty((0, config.max_horizon_steps, len(target_cols))), \
               np.empty((0,))

    feat_arr = df[feature_cols].to_numpy(dtype=np.float32)
    targ_arr = df[target_cols].to_numpy(dtype=np.float32)
    ts_arr   = df["timestamp"].to_numpy()

    W = config.lookback_steps
    H = config.max_horizon_steps
    n = len(df) - W - H + 1   # number of windows that fit
    if n <= 0:
        return np.empty((0, W, len(feature_cols))), \
               np.empty((0, H, len(target_cols))), \
               np.empty((0,))

    X = np.empty((n, W, len(feature_cols)), dtype=np.float32)
    Y = np.empty((n, H, len(target_cols)), dtype=np.float32)
    ts = np.empty(n, dtype=ts_arr.dtype)

    for i in range(n):
        X[i] = feat_arr[i : i + W]
        Y[i] = targ_arr[i + W : i + W + H]
        ts[i] = ts_arr[i + W - 1]   # the "now" timestamp at the end of X
    return X, Y, ts


def build_windows_from_scenarios(
    dfs: List[pd.DataFrame],
    feature_cols: List[str],
    target_cols: List[str],
    config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate window arrays from multiple independent DataFrames
    (e.g. one per scenario). Windows do NOT cross scenario boundaries."""
    Xs, Ys, Ts = [], [], []
    for df in dfs:
        X, Y, ts = build_windows(df, feature_cols, target_cols, config)
        if len(X) > 0:
            Xs.append(X)
            Ys.append(Y)
            Ts.append(ts)
    if not Xs:
        return (np.empty((0, config.lookback_steps, len(feature_cols))),
                np.empty((0, config.max_horizon_steps, len(target_cols))),
                np.empty((0,)))
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), np.concatenate(Ts, axis=0)


def time_split(
    X: np.ndarray, Y: np.ndarray, ts: np.ndarray, val_fraction: float = 0.2
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Time-aware split: last `val_fraction` of the timestamps becomes val.
    No data leakage from future to past."""
    n = len(X)
    if n == 0:
        return ((X, Y), (X, Y))
    n_val = max(1, int(n * val_fraction))
    idx = np.argsort(ts)
    val_idx = idx[-n_val:]
    train_idx = idx[:-n_val]
    return ((X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx]))
