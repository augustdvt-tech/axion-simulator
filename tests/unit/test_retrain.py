"""Tests for scripts/retrain.py — retraining pipeline."""

import sys
import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from scripts.retrain import (
    load_baseline_metrics,
    save_metrics,
    should_promote,
    evaluate_soft_sensor,
    load_train_data,
    retrain_soft_sensor,
    PILOT_PURITY_FEATURES,
    TARGET_PURITY,
    TRAIN_SCENARIOS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh)


def _make_tiny_csv(path: Path, scenario: str, n: int = 20) -> None:
    """Write a minimal CSV with all required feature + target columns."""
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(60, 100, n) for col in PILOT_PURITY_FEATURES}
    data[TARGET_PURITY] = rng.uniform(97, 100, n)
    data["timestamp"] = pd.date_range("2026-01-01", periods=n, freq="1min")
    pd.DataFrame(data).to_csv(path, index=False)


def _make_mock_sensor(mae: float = 0.1) -> MagicMock:
    """Return a sensor mock whose predict_with_confidence returns constant residuals."""
    sensor = MagicMock()

    def _predict(X):
        n = len(X)
        preds = np.full(n, 98.5)
        stds  = np.full(n, 0.05)
        return preds, stds

    sensor.predict_with_confidence.side_effect = _predict

    fit_result = MagicMock()
    fit_result.mae  = mae
    fit_result.rmse = mae * 1.2
    fit_result.r2   = 0.97
    sensor.fit.return_value = fit_result
    return sensor


# ─────────────────────────────────────────────────────────────────────────────
# load_baseline_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadBaselineMetrics:
    def test_returns_none_when_file_absent(self, tmp_path):
        result = load_baseline_metrics(tmp_path / "missing.json")
        assert result is None

    def test_returns_dict_when_file_exists(self, tmp_path):
        metrics = {"mae_holdout": 0.12, "r2_overall": 0.97}
        _write_metrics(tmp_path / "m.json", metrics)
        result = load_baseline_metrics(tmp_path / "m.json")
        assert result == metrics

    def test_preserves_all_keys(self, tmp_path):
        metrics = {"mae_holdout": 0.1, "mae_overall": 0.08, "n_samples": 1000}
        _write_metrics(tmp_path / "m.json", metrics)
        result = load_baseline_metrics(tmp_path / "m.json")
        assert set(result.keys()) == {"mae_holdout", "mae_overall", "n_samples"}


# ─────────────────────────────────────────────────────────────────────────────
# save_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveMetrics:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "m.json"
        save_metrics({"mae": 0.1}, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "m.json"
        save_metrics({"mae": 0.1}, path)
        assert path.exists()

    def test_round_trip(self, tmp_path):
        original = {"mae_holdout": 0.123, "r2_train": 0.98, "n_ensemble": 5}
        path = tmp_path / "m.json"
        save_metrics(original, path)
        loaded = load_baseline_metrics(path)
        assert loaded == original

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "m.json"
        save_metrics({"mae": 0.5}, path)
        save_metrics({"mae": 0.1}, path)
        loaded = load_baseline_metrics(path)
        assert loaded["mae"] == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# should_promote
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldPromote:
    def test_promotes_when_no_baseline(self):
        assert should_promote({"mae_holdout": 0.5}, None) is True

    def test_promotes_when_new_is_better(self):
        assert should_promote({"mae_holdout": 0.10}, {"mae_holdout": 0.12}) is True

    def test_skips_when_new_is_worse(self):
        assert should_promote({"mae_holdout": 0.15}, {"mae_holdout": 0.12}) is False

    def test_promotes_on_equal_with_zero_threshold(self):
        # strictly less than required → equal should NOT promote
        assert should_promote({"mae_holdout": 0.12}, {"mae_holdout": 0.12}) is False

    def test_threshold_blocks_small_improvement(self):
        # 1% improvement, threshold=0.02 → blocked
        assert should_promote(
            {"mae_holdout": 0.119}, {"mae_holdout": 0.12}, threshold=0.02
        ) is False

    def test_threshold_allows_large_improvement(self):
        # 5% improvement, threshold=0.02 → allowed
        assert should_promote(
            {"mae_holdout": 0.114}, {"mae_holdout": 0.12}, threshold=0.02
        ) is True

    def test_promotes_when_new_mae_missing(self):
        assert should_promote({}, {"mae_holdout": 0.12}) is True

    def test_promotes_when_baseline_mae_missing(self):
        assert should_promote({"mae_holdout": 0.10}, {}) is True

    def test_threshold_zero_any_improvement_promotes(self):
        assert should_promote({"mae_holdout": 0.1199}, {"mae_holdout": 0.12},
                               threshold=0.0) is True


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_soft_sensor
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateSoftSensor:
    @pytest.fixture
    def tiny_data(self, tmp_path):
        """Create a small data dir with one train scenario + one holdout."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_tiny_csv(data_dir / "normal.csv", "normal", n=20)
        _make_tiny_csv(data_dir / "sensor_failure.csv", "sensor_failure", n=10)
        return data_dir

    def test_returns_mae_overall(self, tiny_data):
        rng = np.random.default_rng(0)
        data = {col: rng.uniform(60, 100, 20) for col in PILOT_PURITY_FEATURES}
        data[TARGET_PURITY] = rng.uniform(97, 100, 20)
        X = pd.DataFrame(data)[PILOT_PURITY_FEATURES]
        y = pd.Series(data[TARGET_PURITY])
        scn = pd.Series(["normal"] * 20)

        sensor = _make_mock_sensor()
        result = evaluate_soft_sensor(sensor, X, y, scn, tiny_data,
                                      holdout_scenarios=["sensor_failure"],
                                      train_scenarios=["normal"])
        assert "mae_overall" in result
        assert result["mae_overall"] >= 0

    def test_returns_per_scenario_mae(self, tiny_data):
        rng = np.random.default_rng(1)
        data = {col: rng.uniform(60, 100, 20) for col in PILOT_PURITY_FEATURES}
        data[TARGET_PURITY] = rng.uniform(97, 100, 20)
        X = pd.DataFrame(data)[PILOT_PURITY_FEATURES]
        y = pd.Series(data[TARGET_PURITY])
        scn = pd.Series(["normal"] * 20)

        sensor = _make_mock_sensor()
        result = evaluate_soft_sensor(sensor, X, y, scn, tiny_data,
                                      holdout_scenarios=["sensor_failure"],
                                      train_scenarios=["normal"])
        assert "mae_normal" in result

    def test_returns_holdout_mae_when_csv_exists(self, tiny_data):
        rng = np.random.default_rng(2)
        data = {col: rng.uniform(60, 100, 10) for col in PILOT_PURITY_FEATURES}
        data[TARGET_PURITY] = rng.uniform(97, 100, 10)
        X = pd.DataFrame(data)[PILOT_PURITY_FEATURES]
        y = pd.Series(data[TARGET_PURITY])
        scn = pd.Series(["normal"] * 10)

        sensor = _make_mock_sensor()
        result = evaluate_soft_sensor(sensor, X, y, scn, tiny_data,
                                      holdout_scenarios=["sensor_failure"],
                                      train_scenarios=["normal"])
        assert "mae_holdout" in result

    def test_no_holdout_key_when_csv_absent(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rng = np.random.default_rng(3)
        data = {col: rng.uniform(60, 100, 5) for col in PILOT_PURITY_FEATURES}
        data[TARGET_PURITY] = rng.uniform(97, 100, 5)
        X = pd.DataFrame(data)[PILOT_PURITY_FEATURES]
        y = pd.Series(data[TARGET_PURITY])
        scn = pd.Series(["normal"] * 5)

        sensor = _make_mock_sensor()
        result = evaluate_soft_sensor(sensor, X, y, scn, empty_dir,
                                      holdout_scenarios=["sensor_failure"],
                                      train_scenarios=["normal"])
        assert "mae_holdout" not in result


# ─────────────────────────────────────────────────────────────────────────────
# load_train_data
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadTrainData:
    def test_loads_available_scenarios(self, tmp_path):
        _make_tiny_csv(tmp_path / "normal.csv", "normal", n=15)
        _make_tiny_csv(tmp_path / "thermal_drift.csv", "thermal_drift", n=10)
        X, y, scn = load_train_data(tmp_path, scenarios=["normal", "thermal_drift"])
        assert len(X) == 25
        assert set(scn.unique()) == {"normal", "thermal_drift"}

    def test_skips_missing_csvs(self, tmp_path):
        _make_tiny_csv(tmp_path / "normal.csv", "normal", n=10)
        X, y, scn = load_train_data(tmp_path, scenarios=["normal", "missing_scenario"])
        assert len(X) == 10

    def test_raises_when_no_csvs(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_train_data(tmp_path, scenarios=["missing"])

    def test_returns_correct_features(self, tmp_path):
        _make_tiny_csv(tmp_path / "normal.csv", "normal", n=10)
        X, y, scn = load_train_data(tmp_path, scenarios=["normal"])
        assert list(X.columns) == PILOT_PURITY_FEATURES
        assert y.name == TARGET_PURITY


# ─────────────────────────────────────────────────────────────────────────────
# retrain_soft_sensor (integration — PuritySoftSensor mocked for speed)
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrainSoftSensor:
    @pytest.fixture
    def data_dir(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        for s in ["normal", "thermal_drift", "sensor_failure"]:
            _make_tiny_csv(d / f"{s}.csv", s, n=20)
        return d

    @pytest.fixture
    def patch_sensor(self, monkeypatch):
        """Replace PuritySoftSensor with a fast mock in the retrain module."""
        import scripts.retrain as retrain_mod

        class _FakeSensor:
            def __init__(self, n_ensemble=5):
                self.n_ensemble = n_ensemble
                self._mock = _make_mock_sensor(mae=0.08)

            def fit(self, X, y):
                return self._mock.fit(X, y)

            def predict_with_confidence(self, X):
                return self._mock.predict_with_confidence(X)

            def save(self, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("fake-model")

        monkeypatch.setattr(retrain_mod, "PuritySoftSensor", _FakeSensor)

    def test_promotes_on_first_run(self, data_dir, tmp_path, patch_sensor):
        models_dir = tmp_path / "models"
        result = retrain_soft_sensor(
            data_dir=data_dir, models_dir=models_dir,
            force=False, threshold=0.0,
            n_ensemble=2,
        )
        assert result["promoted"] is True
        assert result["baseline_metrics"] is None

    def test_saves_model_and_metrics_on_promote(self, data_dir, tmp_path, patch_sensor):
        models_dir = tmp_path / "models"
        retrain_soft_sensor(
            data_dir=data_dir, models_dir=models_dir,
            force=False, threshold=0.0, n_ensemble=2,
        )
        assert (models_dir / "purity_soft_sensor.joblib").exists()
        assert (models_dir / "purity_soft_sensor.metrics.json").exists()

    def test_promotes_when_better(self, data_dir, tmp_path, patch_sensor):
        models_dir = tmp_path / "models"
        # Baseline with a worse (higher) holdout MAE
        _write_metrics(
            models_dir / "purity_soft_sensor.metrics.json",
            {"mae_holdout": 10.0, "mae_overall": 10.0},
        )
        result = retrain_soft_sensor(
            data_dir=data_dir, models_dir=models_dir,
            force=False, threshold=0.0, n_ensemble=2,
        )
        assert result["promoted"] is True

    def test_skips_when_not_better(self, data_dir, tmp_path, patch_sensor):
        models_dir = tmp_path / "models"
        # Baseline already has perfect MAE — new model (0.08) can't beat 0.0
        _write_metrics(
            models_dir / "purity_soft_sensor.metrics.json",
            {"mae_holdout": 0.0, "mae_overall": 0.0},
        )
        result = retrain_soft_sensor(
            data_dir=data_dir, models_dir=models_dir,
            force=False, threshold=0.0, n_ensemble=2,
        )
        assert result["promoted"] is False

    def test_force_promotes_even_when_not_better(self, data_dir, tmp_path, patch_sensor):
        models_dir = tmp_path / "models"
        _write_metrics(
            models_dir / "purity_soft_sensor.metrics.json",
            {"mae_holdout": 0.0},
        )
        result = retrain_soft_sensor(
            data_dir=data_dir, models_dir=models_dir,
            force=True, threshold=0.0, n_ensemble=2,
        )
        assert result["promoted"] is True

    def test_result_contains_new_metrics(self, data_dir, tmp_path, patch_sensor):
        models_dir = tmp_path / "models"
        result = retrain_soft_sensor(
            data_dir=data_dir, models_dir=models_dir,
            force=False, threshold=0.0, n_ensemble=2,
        )
        assert "mae_overall" in result["new_metrics"]
        assert "n_samples" in result["new_metrics"]
        assert "model_path" in result
