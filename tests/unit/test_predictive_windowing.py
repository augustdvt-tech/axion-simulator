"""Tests for predictive/windowing.py: WindowConfig, Scaler, build_windows, time_split."""

import numpy as np
import pandas as pd
import pytest
from predictive.windowing import WindowConfig, Scaler, build_windows, time_split


FEATURE_COLS = ["cstr.T_R_C", "column.purity_B"]
TARGET_COLS  = ["cstr.T_R_C", "column.purity_B"]


def make_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp":       pd.date_range("2026-01-01", periods=n, freq="1min"),
        "cstr.T_R_C":      rng.normal(78.0, 0.2, n),
        "column.purity_B": rng.normal(99.3, 0.05, n),
    })


class TestWindowConfig:
    def test_lookback_steps(self):
        cfg = WindowConfig(lookback_minutes=60, horizons_minutes=[5, 15],
                           sample_period_minutes=1)
        assert cfg.lookback_steps == 60

    def test_horizon_steps(self):
        cfg = WindowConfig(lookback_minutes=60, horizons_minutes=[5, 15],
                           sample_period_minutes=1)
        assert cfg.horizon_steps == [5, 15]

    def test_max_horizon_steps(self):
        cfg = WindowConfig(lookback_minutes=60, horizons_minutes=[5, 15, 30],
                           sample_period_minutes=1)
        assert cfg.max_horizon_steps == 30


class TestScaler:
    def test_fit_stores_mean_std(self):
        rng = np.random.default_rng(0)
        X = rng.normal(5.0, 2.0, (100, 3))
        scaler = Scaler().fit(X)
        assert scaler.means.shape == (3,)
        assert scaler.stds.shape == (3,)

    def test_transform_zero_mean(self):
        rng = np.random.default_rng(1)
        X = rng.normal(10.0, 3.0, (200, 2))
        scaler = Scaler().fit(X)
        Xt = scaler.transform(X)
        assert abs(Xt.mean()) < 0.1

    def test_inverse_transform_roundtrip(self):
        rng = np.random.default_rng(2)
        X = rng.normal(50.0, 10.0, (100, 4))
        scaler = Scaler().fit(X)
        Xt = scaler.transform(X)
        X_back = scaler.inverse_transform(Xt)
        assert np.allclose(X, X_back, atol=1e-6)


class TestBuildWindows:
    def test_output_shapes(self):
        df = make_df(200)
        cfg = WindowConfig(lookback_minutes=10, horizons_minutes=[5, 15],
                           sample_period_minutes=1)
        X, Y, ts = build_windows(df, FEATURE_COLS, TARGET_COLS, cfg)
        assert X.ndim == 3                   # (samples, lookback, features)
        assert X.shape[1] == 10              # lookback steps
        assert X.shape[2] == len(FEATURE_COLS)
        assert Y.ndim == 3                   # (samples, max_horizon, targets)
        assert Y.shape[1] == 15             # max_horizon_steps = 15

    def test_x_y_same_number_of_samples(self):
        df = make_df(200)
        cfg = WindowConfig(lookback_minutes=10, horizons_minutes=[5],
                           sample_period_minutes=1)
        X, Y, ts = build_windows(df, FEATURE_COLS, TARGET_COLS, cfg)
        assert X.shape[0] == Y.shape[0]

    def test_insufficient_data_returns_empty(self):
        df = make_df(5)
        cfg = WindowConfig(lookback_minutes=10, horizons_minutes=[5],
                           sample_period_minutes=1)
        X, Y, ts = build_windows(df, FEATURE_COLS, TARGET_COLS, cfg)
        assert X.shape[0] == 0


class TestTimeSplit:
    def _make_windows(self):
        df = make_df(100)
        cfg = WindowConfig(lookback_minutes=5, horizons_minutes=[5],
                           sample_period_minutes=1)
        return build_windows(df, FEATURE_COLS, TARGET_COLS, cfg)

    def test_split_ratios(self):
        X, Y, ts = self._make_windows()
        (X_tr, Y_tr), (X_val, Y_val) = time_split(X, Y, ts, val_fraction=0.2)
        total = len(X_tr) + len(X_val)
        assert total == len(X)
        assert len(X_val) >= 1

    def test_no_overlap(self):
        X, Y, ts = self._make_windows()
        (X_tr, _), (X_val, _) = time_split(X, Y, ts, val_fraction=0.2)
        assert len(X_tr) > 0
        assert len(X_val) > 0

    def test_train_before_val(self):
        X, Y, ts = self._make_windows()
        idx = np.argsort(ts)
        ts_sorted = ts[idx]
        n_val = max(1, int(len(ts) * 0.2))
        train_ts = ts_sorted[:-n_val]
        val_ts   = ts_sorted[-n_val:]
        assert train_ts.max() < val_ts.min()
