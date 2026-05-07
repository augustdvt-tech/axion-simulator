"""Tests for axion_mlflow.py — MLflow tracking wrapper."""

import sys
import warnings
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, ".")

import axion_mlflow
from axion_mlflow import Run


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_mock_mlflow():
    """Return a MagicMock that mimics the mlflow public API."""
    m = MagicMock()
    return m


# ──────────────────────────────────────────────────────────────────────────────
# is_available()
# ──────────────────────────────────────────────────────────────────────────────

class TestIsAvailable:
    def test_reflects_module_flag_true(self, monkeypatch):
        monkeypatch.setattr(axion_mlflow, "_AVAILABLE", True)
        assert axion_mlflow.is_available() is True

    def test_reflects_module_flag_false(self, monkeypatch):
        monkeypatch.setattr(axion_mlflow, "_AVAILABLE", False)
        assert axion_mlflow.is_available() is False


# ──────────────────────────────────────────────────────────────────────────────
# Run — unavailable MLflow (graceful no-ops)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunUnavailable:
    @pytest.fixture(autouse=True)
    def _patch_unavailable(self, monkeypatch):
        monkeypatch.setattr(axion_mlflow, "_AVAILABLE", False)

    def test_enter_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with Run("test-exp"):
                pass
        assert len(w) == 1
        assert "MLflow not installed" in str(w[0].message)

    def test_active_is_false_when_unavailable(self):
        run = Run("test-exp")
        with run:
            assert run._active is False

    def test_log_params_is_noop(self):
        with Run("test-exp") as run:
            run.log_params({"k": 1})  # must not raise

    def test_log_metric_is_noop(self):
        with Run("test-exp") as run:
            run.log_metric("mae", 0.1)

    def test_log_metrics_is_noop(self):
        with Run("test-exp") as run:
            run.log_metrics({"mae": 0.1, "r2": 0.9})

    def test_log_epoch_metrics_is_noop(self):
        with Run("test-exp") as run:
            run.log_epoch_metrics([0.5, 0.3], [0.6, 0.4])

    def test_set_tags_is_noop(self):
        with Run("test-exp") as run:
            run.set_tags({"version": "1"})

    def test_log_artifact_is_noop(self):
        with Run("test-exp") as run:
            run.log_artifact("/tmp/model.joblib")

    def test_log_artifacts_is_noop(self):
        with Run("test-exp") as run:
            run.log_artifacts("/tmp/models")


# ──────────────────────────────────────────────────────────────────────────────
# Run — available MLflow (delegates to mlflow module)
# ──────────────────────────────────────────────────────────────────────────────

class TestRunAvailable:
    @pytest.fixture(autouse=True)
    def _patch_available(self, monkeypatch):
        monkeypatch.setattr(axion_mlflow, "_AVAILABLE", True)
        self.mlflow = _make_mock_mlflow()
        # raising=False because mlflow attr may not exist when package is absent
        monkeypatch.setattr(axion_mlflow, "mlflow", self.mlflow, raising=False)

    def test_sets_experiment_on_enter(self):
        with Run("my-exp"):
            pass
        self.mlflow.set_experiment.assert_called_once_with("my-exp")

    def test_starts_run_with_name(self):
        with Run("my-exp", run_name="v1"):
            pass
        self.mlflow.start_run.assert_called_once_with(run_name="v1")

    def test_sets_tracking_uri_when_provided(self):
        with Run("my-exp", tracking_uri="http://localhost:5000"):
            pass
        self.mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_no_tracking_uri_when_not_provided(self):
        with Run("my-exp"):
            pass
        self.mlflow.set_tracking_uri.assert_not_called()

    def test_ends_run_on_exit(self):
        with Run("my-exp"):
            pass
        self.mlflow.end_run.assert_called_once()

    def test_active_true_inside_context(self):
        with Run("my-exp") as run:
            assert run._active is True

    def test_active_false_after_exit(self):
        run = Run("my-exp")
        with run:
            pass
        assert run._active is False

    def test_log_param_delegates(self):
        with Run("my-exp") as run:
            run.log_param("key", "val")
        self.mlflow.log_param.assert_called_once_with("key", "val")

    def test_log_params_delegates(self):
        with Run("my-exp") as run:
            run.log_params({"a": 1, "b": 2})
        self.mlflow.log_params.assert_called_once_with({"a": 1, "b": 2})

    def test_log_metric_delegates(self):
        with Run("my-exp") as run:
            run.log_metric("mae", 0.1, step=1)
        self.mlflow.log_metric.assert_called_once_with("mae", 0.1, step=1)

    def test_log_metrics_delegates(self):
        with Run("my-exp") as run:
            run.log_metrics({"mae": 0.1})
        self.mlflow.log_metrics.assert_called_once_with({"mae": 0.1})

    def test_log_epoch_metrics_steps(self):
        train_loss = [0.5, 0.3, 0.2]
        val_loss   = [0.6, 0.4, 0.3]
        with Run("my-exp") as run:
            run.log_epoch_metrics(train_loss, val_loss)
        calls = self.mlflow.log_metrics.call_args_list
        assert len(calls) == 3
        assert calls[0] == call({"train_loss": 0.5, "val_loss": 0.6}, step=1)
        assert calls[2] == call({"train_loss": 0.2, "val_loss": 0.3}, step=3)

    def test_set_tag_delegates(self):
        with Run("my-exp") as run:
            run.set_tag("env", "prod")
        self.mlflow.set_tag.assert_called_once_with("env", "prod")

    def test_set_tags_delegates(self):
        with Run("my-exp") as run:
            run.set_tags({"a": "1"})
        self.mlflow.set_tags.assert_called_once_with({"a": "1"})

    def test_log_artifact_delegates(self):
        with Run("my-exp") as run:
            run.log_artifact("/tmp/model.joblib")
        self.mlflow.log_artifact.assert_called_once_with("/tmp/model.joblib")

    def test_log_artifacts_delegates(self):
        with Run("my-exp") as run:
            run.log_artifacts("/tmp/models", artifact_path="model")
        self.mlflow.log_artifacts.assert_called_once_with(
            "/tmp/models", artifact_path="model"
        )

    def test_exception_in_start_run_does_not_raise(self):
        self.mlflow.start_run.side_effect = RuntimeError("connection refused")
        with Run("my-exp") as run:
            assert run._active is False  # gracefully degraded

    def test_exception_in_end_run_does_not_raise(self):
        self.mlflow.end_run.side_effect = RuntimeError("boom")
        with Run("my-exp"):
            pass  # must not propagate

    def test_no_warning_when_available(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with Run("my-exp"):
                pass
        assert len(w) == 0
