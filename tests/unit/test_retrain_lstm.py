"""Tests for the LSTM retraining pipeline in scripts/retrain.py."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

sys.path.insert(0, ".")

from scripts import retrain as retrain_mod
from scripts.retrain import (
    aggregate_lstm_metrics,
    should_promote_lstm,
    load_lstm_train_data,
    retrain_lstm,
    LSTM_FEATURE_COLS,
    LSTM_TARGET_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FakeLSTMMetrics:
    by_horizon: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    train_loss_history: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.3])
    val_loss_history:   List[float] = field(default_factory=lambda: [1.1, 0.6, 0.4])
    n_train: int = 1000
    n_val:   int = 200


def _make_metrics_obj(mae_value: float = 0.5) -> _FakeLSTMMetrics:
    """Build a metrics object with 4 horizons × 2 targets."""
    by_horizon = {}
    for h in (5, 15, 30, 60):
        by_horizon[h] = {
            "cstr.T_R_C":      {"mae": mae_value, "rmse": mae_value * 1.3, "r2": 0.9},
            "column.purity_B": {"mae": mae_value, "rmse": mae_value * 1.3, "r2": 0.9},
        }
    return _FakeLSTMMetrics(by_horizon=by_horizon)


class _FakeForecaster:
    """In-memory stand-in for LSTMForecaster — no TF required."""
    last_instance = None

    def __init__(self, feature_cols, target_cols, config, **kwargs):
        self.feature_cols = feature_cols
        self.target_cols  = target_cols
        self.config       = config
        self.fit_called_with: Dict[str, Any] = {}
        self.saved_to: Path | None = None
        _FakeForecaster.last_instance = self

    def fit(self, scenario_dfs, epochs=30, batch_size=64, verbose=0, val_fraction=0.2):
        self.fit_called_with = {
            "n_scenarios": len(scenario_dfs),
            "epochs":      epochs,
            "batch_size":  batch_size,
            "val_fraction": val_fraction,
        }
        return _make_metrics_obj(mae_value=0.4)

    def save(self, path_dir):
        path_dir = Path(path_dir)
        path_dir.mkdir(parents=True, exist_ok=True)
        (path_dir / "model.keras").write_text("fake-keras-model")
        self.saved_to = path_dir


@pytest.fixture
def patch_forecaster(monkeypatch):
    """Make `from predictive import LSTMForecaster` resolve to the fake."""
    import sys as _sys
    import types

    @dataclass
    class _FakeWindowConfig:
        lookback_minutes: int = 120
        horizons_minutes: list = field(default_factory=lambda: [5, 15, 30, 60])
        sample_period_minutes: int = 1

    fake_predictive = types.SimpleNamespace(LSTMForecaster=_FakeForecaster)
    fake_windowing  = types.SimpleNamespace(WindowConfig=_FakeWindowConfig)

    monkeypatch.setitem(_sys.modules, "predictive", fake_predictive)
    monkeypatch.setitem(_sys.modules, "predictive.windowing", fake_windowing)
    yield


def _write_csv(path: Path, n_rows: int = 30) -> None:
    """Write a minimal scenario CSV with all LSTM feature columns."""
    base = pd.Timestamp("2026-01-01")
    cols = set(LSTM_FEATURE_COLS) | set(LSTM_TARGET_COLS)
    data = {"timestamp": [base + pd.Timedelta(minutes=i) for i in range(n_rows)]}
    for c in cols:
        data[c] = [1.0 + 0.01 * i for i in range(n_rows)]
    pd.DataFrame(data).to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_lstm_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateLstmMetrics:
    def test_empty_by_horizon_returns_minimal(self):
        m = _FakeLSTMMetrics(by_horizon={})
        flat = aggregate_lstm_metrics(m)
        assert flat["n_train"] == 1000
        assert "mae_overall" not in flat   # no horizons → no aggregate

    def test_mae_overall_is_average(self):
        m = _make_metrics_obj(mae_value=0.5)
        flat = aggregate_lstm_metrics(m)
        # all 8 entries are 0.5 → mean is 0.5
        assert flat["mae_overall"] == pytest.approx(0.5)

    def test_per_pair_keys_present(self):
        m = _make_metrics_obj(mae_value=0.5)
        flat = aggregate_lstm_metrics(m)
        assert "mae_5min_cstr_T_R_C" in flat
        assert "mae_60min_column_purity_B" in flat

    def test_dots_replaced_with_underscores(self):
        m = _make_metrics_obj(mae_value=0.5)
        flat = aggregate_lstm_metrics(m)
        # Original target name has a dot — flat keys must not
        assert "mae_5min_cstr.T_R_C" not in flat

    def test_mae_worst_is_max(self):
        # craft uneven MAEs
        m = _FakeLSTMMetrics(by_horizon={
            5:  {"x": {"mae": 0.1, "rmse": 0.2, "r2": 0.9}},
            60: {"x": {"mae": 0.9, "rmse": 1.0, "r2": 0.5}},
        })
        flat = aggregate_lstm_metrics(m)
        assert flat["mae_worst"] == pytest.approx(0.9)
        assert flat["mae_overall"] == pytest.approx(0.5)

    def test_n_train_n_val_propagated(self):
        m = _make_metrics_obj()
        flat = aggregate_lstm_metrics(m)
        assert flat["n_train"] == 1000
        assert flat["n_val"]   == 200


# ─────────────────────────────────────────────────────────────────────────────
# should_promote_lstm
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldPromoteLstm:
    def test_no_baseline_promotes(self):
        assert should_promote_lstm({"mae_overall": 0.5}, None) is True

    def test_missing_keys_promotes(self):
        assert should_promote_lstm({}, {"mae_overall": 0.5}) is True
        assert should_promote_lstm({"mae_overall": 0.5}, {}) is True

    def test_strict_improvement_promotes(self):
        assert should_promote_lstm(
            {"mae_overall": 0.4}, {"mae_overall": 0.5}, threshold=0.0,
        ) is True

    def test_equal_does_not_promote(self):
        assert should_promote_lstm(
            {"mae_overall": 0.5}, {"mae_overall": 0.5}, threshold=0.0,
        ) is False

    def test_worse_does_not_promote(self):
        assert should_promote_lstm(
            {"mae_overall": 0.6}, {"mae_overall": 0.5}, threshold=0.0,
        ) is False

    def test_threshold_blocks_marginal(self):
        # 1% improvement, threshold demands 5%
        assert should_promote_lstm(
            {"mae_overall": 0.495}, {"mae_overall": 0.5}, threshold=0.05,
        ) is False

    def test_threshold_allows_clear_win(self):
        # 20% improvement, threshold 5%
        assert should_promote_lstm(
            {"mae_overall": 0.40}, {"mae_overall": 0.5}, threshold=0.05,
        ) is True


# ─────────────────────────────────────────────────────────────────────────────
# load_lstm_train_data
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLstmTrainData:
    def test_loads_present_csvs(self, tmp_path):
        _write_csv(tmp_path / "normal.csv")
        _write_csv(tmp_path / "thermal_drift.csv")
        dfs = load_lstm_train_data(tmp_path, scenarios=["normal", "thermal_drift"])
        assert len(dfs) == 2

    def test_skips_missing_csvs(self, tmp_path):
        _write_csv(tmp_path / "normal.csv")
        # thermal_drift.csv intentionally missing
        dfs = load_lstm_train_data(tmp_path, scenarios=["normal", "thermal_drift"])
        assert len(dfs) == 1

    def test_raises_when_all_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lstm_train_data(tmp_path, scenarios=["nope"])

    def test_parses_timestamp_column(self, tmp_path):
        _write_csv(tmp_path / "normal.csv")
        dfs = load_lstm_train_data(tmp_path, scenarios=["normal"])
        assert pd.api.types.is_datetime64_any_dtype(dfs[0]["timestamp"])


# ─────────────────────────────────────────────────────────────────────────────
# retrain_lstm — orchestration with mocked forecaster
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrainLstm:
    def test_first_run_promotes(self, tmp_path, patch_forecaster):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _write_csv(data_dir / "normal.csv")
        lstm_dir = tmp_path / "lstm"

        result = retrain_lstm(
            data_dir=data_dir, lstm_dir=lstm_dir,
            force=False, threshold=0.0,
            epochs=2, batch_size=8,
        )

        assert result["promoted"] is True
        assert result["baseline_metrics"] is None
        assert (lstm_dir / "model.keras").exists()
        assert (lstm_dir / "metrics.json").exists()

    def test_metrics_file_has_overall_mae(self, tmp_path, patch_forecaster):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _write_csv(data_dir / "normal.csv")
        lstm_dir = tmp_path / "lstm"

        retrain_lstm(data_dir=data_dir, lstm_dir=lstm_dir,
                     force=False, epochs=1, batch_size=4)
        with open(lstm_dir / "metrics.json") as fh:
            persisted = json.load(fh)
        assert "mae_overall" in persisted

    def test_force_promotes_even_when_worse(self, tmp_path, patch_forecaster):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _write_csv(data_dir / "normal.csv")
        lstm_dir = tmp_path / "lstm"; lstm_dir.mkdir()
        # Plant a much-better baseline so a normal run wouldn't promote
        (lstm_dir / "metrics.json").write_text(json.dumps({"mae_overall": 0.001}))

        result = retrain_lstm(data_dir=data_dir, lstm_dir=lstm_dir,
                              force=True, epochs=1, batch_size=4)
        assert result["promoted"] is True

    def test_no_improvement_does_not_promote(self, tmp_path, patch_forecaster):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _write_csv(data_dir / "normal.csv")
        lstm_dir = tmp_path / "lstm"; lstm_dir.mkdir()
        # Baseline is already as good as the fake fit returns (0.4)
        (lstm_dir / "metrics.json").write_text(json.dumps({"mae_overall": 0.4}))
        baseline_mtime = (lstm_dir / "metrics.json").stat().st_mtime

        result = retrain_lstm(data_dir=data_dir, lstm_dir=lstm_dir,
                              force=False, threshold=0.0,
                              epochs=1, batch_size=4)
        assert result["promoted"] is False
        # Metrics file untouched
        assert (lstm_dir / "metrics.json").stat().st_mtime == baseline_mtime
        # Model file should NOT have been written
        assert not (lstm_dir / "model.keras").exists()

    def test_passes_epochs_and_batch_to_fit(self, tmp_path, patch_forecaster):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _write_csv(data_dir / "normal.csv")
        lstm_dir = tmp_path / "lstm"

        retrain_lstm(data_dir=data_dir, lstm_dir=lstm_dir,
                     epochs=7, batch_size=33, force=True)

        called = _FakeForecaster.last_instance.fit_called_with
        assert called["epochs"] == 7
        assert called["batch_size"] == 33

    def test_returns_baseline_metrics_when_present(self, tmp_path, patch_forecaster):
        data_dir = tmp_path / "data"; data_dir.mkdir()
        _write_csv(data_dir / "normal.csv")
        lstm_dir = tmp_path / "lstm"; lstm_dir.mkdir()
        baseline = {"mae_overall": 0.6}
        (lstm_dir / "metrics.json").write_text(json.dumps(baseline))

        result = retrain_lstm(data_dir=data_dir, lstm_dir=lstm_dir,
                              force=False, threshold=0.0,
                              epochs=1, batch_size=4)
        assert result["baseline_metrics"] == baseline
