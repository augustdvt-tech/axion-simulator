"""Tests for predictive/lstm.py: LSTMForecaster (slow — trains a neural network)."""

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not available in this environment")

from predictive.lstm import LSTMForecaster
from predictive.windowing import WindowConfig, build_windows


pytestmark = pytest.mark.slow

FEATURES = ["cstr.T_R_C", "column.purity_B"]
TARGETS  = ["cstr.T_R_C", "column.purity_B"]


def make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp":       pd.date_range("2026-01-01", periods=n, freq="1min"),
        "cstr.T_R_C":      rng.normal(78.0, 0.3, n),
        "column.purity_B": rng.normal(99.3, 0.1, n),
    })


@pytest.fixture(scope="module")
def trained_forecaster():
    df = make_df(300)
    cfg = WindowConfig(lookback_minutes=10, horizons_minutes=[5, 15],
                       sample_period_minutes=1)
    X, y = build_windows(df, cfg, feature_cols=FEATURES, target_cols=TARGETS)
    forecaster = LSTMForecaster(
        config=cfg,
        feature_cols=FEATURES,
        target_cols=TARGETS,
        hidden_units=[16, 8],
    )
    forecaster.fit(X, y, epochs=3, batch_size=32, verbose=0)
    return forecaster


class TestLSTMForecaster:
    def test_fit_completes(self, trained_forecaster):
        assert trained_forecaster.is_fitted

    def test_predict_at_horizons_returns_dict(self, trained_forecaster):
        df = make_df(50)
        result = trained_forecaster.predict_at_horizons(df, horizons_minutes=[5, 15])
        assert isinstance(result, dict)
        for target in TARGETS:
            assert target in result

    def test_predict_values_physically_plausible(self, trained_forecaster):
        df = make_df(50)
        result = trained_forecaster.predict_at_horizons(df, horizons_minutes=[5])
        for target, horizons in result.items():
            for h, val in horizons.items():
                assert isinstance(val, float)
                assert not np.isnan(val)
