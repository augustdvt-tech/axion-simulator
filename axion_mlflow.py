"""
Axion AI - MLflow Tracking Utilities
=====================================

Thin wrapper around MLflow that degrades gracefully when MLflow is not
installed or not configured. The `Run` context manager is the public API:

    from axion_mlflow import Run

    with Run("axion-soft-sensor", run_name="gbrens_v1") as run:
        run.log_params({"n_estimators": 100, "n_ensemble": 5})
        run.log_metrics({"mae": 0.12, "r2": 0.97})
        run.log_artifact("/path/to/model.joblib")

If MLflow is not installed all calls are silent no-ops — training scripts
continue to work without change. A single warning is emitted at run start.

Configuration:
  MLFLOW_TRACKING_URI  Remote server URL or local path (default: ./mlruns).
                       Leave unset for local file-based tracking.

Typical values:
  MLFLOW_TRACKING_URI=http://localhost:5000      # MLflow server
  MLFLOW_TRACKING_URI=sqlite:///mlruns.db        # SQLite (single-user)
  MLFLOW_TRACKING_URI=./mlruns                   # default (file store)
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

try:
    import mlflow
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def is_available() -> bool:
    """Return True if MLflow is installed."""
    return _AVAILABLE


class Run:
    """Context manager that wraps a single MLflow run.

    All methods are safe to call even when MLflow is unavailable —
    they silently become no-ops so training scripts never crash.
    """

    def __init__(
        self,
        experiment: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self._experiment   = experiment
        self._run_name     = run_name
        self._tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self._active       = False

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "Run":
        if not _AVAILABLE:
            warnings.warn(
                "MLflow not installed — run metrics will not be tracked. "
                "pip install mlflow",
                stacklevel=2,
            )
            return self
        try:
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment)
            mlflow.start_run(run_name=self._run_name)
            self._active = True
        except Exception as exc:
            warnings.warn(f"MLflow run could not start: {exc}", stacklevel=2)
        return self

    def __exit__(self, *_) -> None:
        if self._active:
            try:
                mlflow.end_run()
            except Exception:
                pass
            self._active = False

    # ------------------------------------------------------------------ #
    # Params                                                               #
    # ------------------------------------------------------------------ #

    def log_param(self, key: str, value: Any) -> None:
        if self._active:
            try:
                mlflow.log_param(key, value)
            except Exception:
                pass

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._active:
            try:
                mlflow.log_params(params)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def log_metric(self, key: str, value: float,
                   step: Optional[int] = None) -> None:
        if self._active:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if self._active:
            try:
                mlflow.log_metrics(metrics)
            except Exception:
                pass

    def log_epoch_metrics(
        self,
        train_loss: List[float],
        val_loss: List[float],
    ) -> None:
        """Log per-epoch loss curves as stepped MLflow metrics."""
        if not self._active:
            return
        for epoch, (t, v) in enumerate(zip(train_loss, val_loss), start=1):
            try:
                mlflow.log_metrics({"train_loss": t, "val_loss": v}, step=epoch)
            except Exception:
                break

    # ------------------------------------------------------------------ #
    # Tags                                                                 #
    # ------------------------------------------------------------------ #

    def set_tag(self, key: str, value: str) -> None:
        if self._active:
            try:
                mlflow.set_tag(key, value)
            except Exception:
                pass

    def set_tags(self, tags: Dict[str, str]) -> None:
        if self._active:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Artifacts                                                            #
    # ------------------------------------------------------------------ #

    def log_artifact(self, local_path: str) -> None:
        """Log a single file as an artifact."""
        if self._active:
            try:
                mlflow.log_artifact(local_path)
            except Exception:
                pass

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log all files in a directory as artifacts."""
        if self._active:
            try:
                mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            except Exception:
                pass
