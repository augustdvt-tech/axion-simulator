"""
Axion AI — Retraining Pipeline
================================

Compares a freshly trained model against the current production model and
promotes it if the holdout MAE improves.

Usage:
    python scripts/retrain.py [--model soft_sensor]
                              [--data-dir data/]
                              [--force]
                              [--threshold 0.02]  # require ≥2% MAE improvement

The production model lives at results/models/purity_soft_sensor.joblib.
Its evaluation metrics are persisted alongside it in
results/models/purity_soft_sensor.metrics.json so they survive process restarts.

On the first run (no metrics file) the new model is always promoted.
For subsequent runs, promotion requires:
    new_mae_holdout < baseline_mae_holdout * (1 - threshold)
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from soft_sensor import PuritySoftSensor, PILOT_PURITY_FEATURES, TARGET_PURITY
from axion_mlflow import Run
from axion_logging import get_logger
from data_versioning import take_snapshot

log = get_logger("retrain")

DATA_DIR    = Path(__file__).parent.parent / "data"
MODELS_DIR  = Path(__file__).parent.parent / "results" / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TRAIN_SCENARIOS = [
    "normal", "thermal_drift", "feed_perturbation",
    "reactor_instability", "quality_degradation",
    "energy_waste", "product_grade_change",
]
HOLDOUT_SCENARIOS = ["sensor_failure"]

SOFT_SENSOR_MODEL_PATH   = MODELS_DIR / "purity_soft_sensor.joblib"
SOFT_SENSOR_METRICS_PATH = MODELS_DIR / "purity_soft_sensor.metrics.json"

LSTM_DIR          = MODELS_DIR / "lstm_forecaster"
LSTM_METRICS_PATH = LSTM_DIR / "metrics.json"

LSTM_FEATURE_COLS = [
    "cstr.T_R_C", "column.purity_B", "column.Q_reb_kW", "cstr.conversion",
    "column.RR", "cstr.F_cool", "cstr.F_feed",
    "cstr.C_A", "cstr.T_feed_C",
    "column.T_bot_C", "column.T_top_C", "cstr.T_J_C",
]
LSTM_TARGET_COLS = [
    "cstr.T_R_C", "column.purity_B", "column.Q_reb_kW", "cstr.conversion",
]
LSTM_HORIZONS_MINUTES = [5, 15, 30, 60]
LSTM_LOOKBACK_MINUTES = 120


# ─────────────────────────────────────────────────────────────────────────────
# Pure utility functions (fully unit-testable, no I/O side-effects on data)
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """Return metrics from the current production model, or None if absent."""
    if not metrics_path.exists():
        return None
    with open(metrics_path) as fh:
        return json.load(fh)


def save_metrics(metrics: Dict[str, Any], metrics_path: Path) -> None:
    """Persist metrics JSON next to the model file."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def should_promote(
    new_metrics: Dict[str, float],
    baseline: Optional[Dict[str, float]],
    threshold: float = 0.0,
) -> bool:
    """Decide whether the new model should replace the current production model.

    Promotion rules (in order):
      1. No baseline exists → always promote (first run).
      2. Holdout MAE key missing in either dict → promote conservatively.
      3. new_mae_holdout < baseline_mae_holdout * (1 - threshold) → promote.
    """
    if baseline is None:
        return True
    new_mae      = new_metrics.get("mae_holdout")
    baseline_mae = baseline.get("mae_holdout")
    if new_mae is None or baseline_mae is None:
        return True
    return float(new_mae) < float(baseline_mae) * (1.0 - float(threshold))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_train_data(
    data_dir: Path,
    scenarios: list[str] = TRAIN_SCENARIOS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble combined X / y / scenario-label series from scenario CSVs."""
    frames = []
    for s in scenarios:
        path = data_dir / f"{s}.csv"
        if not path.exists():
            log.warning("scenario_csv_missing", extra={"scenario": s, "path": str(path)})
            continue
        df = pd.read_csv(path)
        df["_scenario"] = s
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No scenario CSVs found in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    needed = PILOT_PURITY_FEATURES + [TARGET_PURITY]
    mask = combined[needed].notna().all(axis=1)
    combined = combined.loc[mask].reset_index(drop=True)

    X   = combined[PILOT_PURITY_FEATURES].copy()
    y   = combined[TARGET_PURITY].copy()
    scn = combined["_scenario"].copy()
    return X, y, scn


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_soft_sensor(
    sensor: PuritySoftSensor,
    X: pd.DataFrame,
    y: pd.Series,
    scn: pd.Series,
    data_dir: Path,
    holdout_scenarios: list[str] = HOLDOUT_SCENARIOS,
    train_scenarios: list[str] = TRAIN_SCENARIOS,
) -> Dict[str, float]:
    """Compute overall, per-scenario, and holdout MAE/RMSE/R²."""
    preds, _ = sensor.predict_with_confidence(X)
    residuals = y.values - preds

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y.values - float(y.mean())) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics: Dict[str, float] = {
        "mae_overall":  float(np.mean(np.abs(residuals))),
        "rmse_overall": float(np.sqrt(np.mean(residuals ** 2))),
        "r2_overall":   r2,
    }

    for s in train_scenarios:
        idx = scn.values == s
        if idx.any():
            res_s = y.values[idx] - preds[idx]
            metrics[f"mae_{s}"] = float(np.mean(np.abs(res_s)))

    for s in holdout_scenarios:
        path = data_dir / f"{s}.csv"
        if not path.exists():
            continue
        df_h = pd.read_csv(path)
        mask = df_h[PILOT_PURITY_FEATURES + [TARGET_PURITY]].notna().all(axis=1)
        if not mask.any():
            continue
        Xh = df_h.loc[mask, PILOT_PURITY_FEATURES]
        yh = df_h.loc[mask, TARGET_PURITY]
        ph, _ = sensor.predict_with_confidence(Xh)
        res_h = yh.values - ph
        metrics["mae_holdout"]  = float(np.mean(np.abs(res_h)))
        metrics["rmse_holdout"] = float(np.sqrt(np.mean(res_h ** 2)))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def retrain_soft_sensor(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    force: bool = False,
    threshold: float = 0.0,
    n_ensemble: int = 5,
) -> Dict[str, Any]:
    """Full retrain → evaluate → promote cycle for the purity soft sensor.

    Returns:
        promoted        bool   — whether the new model was written to disk
        new_metrics     dict   — evaluation metrics for the newly trained model
        baseline_metrics dict | None — metrics of the model that was on disk
        model_path      str    — path to the (possibly promoted) model
    """
    model_path   = models_dir / "purity_soft_sensor.joblib"
    metrics_path = models_dir / "purity_soft_sensor.metrics.json"

    baseline = load_baseline_metrics(metrics_path)
    log.info("baseline_loaded", extra={"has_baseline": baseline is not None})

    X, y, scn = load_train_data(data_dir)
    log.info("data_loaded", extra={"n_samples": len(X), "n_scenarios": int(scn.nunique())})

    sensor = PuritySoftSensor(n_ensemble=n_ensemble)
    fit_result = sensor.fit(X, y)
    log.info("fit_complete", extra={"mae_train": fit_result.mae, "r2_train": fit_result.r2})

    new_metrics = evaluate_soft_sensor(sensor, X, y, scn, data_dir)
    new_metrics.update({
        "mae_train":  fit_result.mae,
        "rmse_train": fit_result.rmse,
        "r2_train":   fit_result.r2,
        "n_samples":  len(X),
        "n_ensemble": n_ensemble,
    })

    # Stamp the trained model with the data snapshot it saw — gives us full
    # reproducibility ("which CSVs was this model trained on?")
    try:
        snap = take_snapshot(
            data_dir=data_dir,
            snapshots_dir=data_dir / ".versions",
            message=f"auto: soft_sensor retrain at {fit_result.r2:.3f} R²",
        )
        new_metrics["data_snapshot_id"] = snap.snapshot_id
        new_metrics["data_snapshot_files"] = len(snap.files)
        log.info("data_snapshot_taken",
                 extra={"snapshot_id": snap.snapshot_id, "n_files": len(snap.files)})
    except Exception as e:
        log.warning("data_snapshot_failed", extra={"error": str(e)})

    promoted = force or should_promote(new_metrics, baseline, threshold)

    if promoted:
        models_dir.mkdir(parents=True, exist_ok=True)
        sensor.save(model_path)
        save_metrics(new_metrics, metrics_path)
        log.info("model_promoted", extra={"model_path": str(model_path)})
    else:
        log.info("model_not_promoted", extra={
            "new_mae_holdout":      new_metrics.get("mae_holdout"),
            "baseline_mae_holdout": baseline.get("mae_holdout") if baseline else None,
        })

    return {
        "promoted":        promoted,
        "new_metrics":     new_metrics,
        "baseline_metrics": baseline,
        "model_path":      str(model_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LSTM retraining
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_lstm_metrics(metrics_obj: Any) -> Dict[str, float]:
    """Flatten the nested per-horizon, per-target LSTM metrics into a flat dict.

    The forecaster returns metrics shaped as:
        by_horizon[horizon_min][target_col] = {"mae", "rmse", "r2"}

    For promotion we collapse to a single scalar `mae_overall` (mean across
    all horizon/target pairs), while keeping per-pair MAEs for inspection.
    """
    flat: Dict[str, float] = {
        "n_train":  int(getattr(metrics_obj, "n_train", 0)),
        "n_val":    int(getattr(metrics_obj, "n_val", 0)),
    }
    by_horizon = getattr(metrics_obj, "by_horizon", {}) or {}
    all_maes: list[float] = []
    for h_min, by_tgt in by_horizon.items():
        for tgt, m in by_tgt.items():
            tgt_safe = tgt.replace(".", "_")
            mae = float(m.get("mae", float("nan")))
            r2  = float(m.get("r2",  float("nan")))
            flat[f"mae_{h_min}min_{tgt_safe}"] = mae
            flat[f"r2_{h_min}min_{tgt_safe}"]  = r2
            if not np.isnan(mae):
                all_maes.append(mae)
    if all_maes:
        flat["mae_overall"] = float(np.mean(all_maes))
        flat["mae_worst"]   = float(np.max(all_maes))
    return flat


def should_promote_lstm(
    new_metrics: Dict[str, float],
    baseline: Optional[Dict[str, float]],
    threshold: float = 0.0,
) -> bool:
    """Promote when `mae_overall` improves by at least `threshold` (relative).

    Same shape as `should_promote` for the soft sensor, but keyed off
    `mae_overall` since the LSTM uses a time-split internal validation.
    """
    if baseline is None:
        return True
    new_mae      = new_metrics.get("mae_overall")
    baseline_mae = baseline.get("mae_overall")
    if new_mae is None or baseline_mae is None:
        return True
    return float(new_mae) < float(baseline_mae) * (1.0 - float(threshold))


def load_lstm_train_data(
    data_dir: Path,
    scenarios: list[str] = TRAIN_SCENARIOS,
) -> list[pd.DataFrame]:
    """Load each scenario CSV as a DataFrame with parsed timestamps."""
    out: list[pd.DataFrame] = []
    for s in scenarios:
        path = data_dir / f"{s}.csv"
        if not path.exists():
            log.warning("scenario_csv_missing",
                        extra={"scenario": s, "path": str(path)})
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        out.append(df)
    if not out:
        raise FileNotFoundError(f"No scenario CSVs found in {data_dir}")
    return out


def retrain_lstm(
    data_dir: Path = DATA_DIR,
    lstm_dir: Path = LSTM_DIR,
    force: bool = False,
    threshold: float = 0.0,
    epochs: int = 30,
    batch_size: int = 64,
    val_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Full retrain → evaluate → promote cycle for the LSTM forecaster.

    TensorFlow / Keras are imported lazily here so the rest of the retrain
    script (and the soft sensor pipeline) keep working when TF is absent.
    """
    try:
        from predictive import LSTMForecaster
        from predictive.windowing import WindowConfig
    except ImportError as e:
        raise RuntimeError(
            f"LSTM retrain requires TensorFlow: {e}. "
            f"Install with `pip install tensorflow>=2.15`."
        ) from e

    metrics_path = lstm_dir / "metrics.json"
    baseline = load_baseline_metrics(metrics_path)
    log.info("lstm_baseline_loaded", extra={"has_baseline": baseline is not None})

    scenario_dfs = load_lstm_train_data(data_dir)
    log.info("lstm_data_loaded", extra={
        "n_scenarios": len(scenario_dfs),
        "total_rows":  int(sum(len(d) for d in scenario_dfs)),
    })

    config = WindowConfig(
        lookback_minutes=LSTM_LOOKBACK_MINUTES,
        horizons_minutes=LSTM_HORIZONS_MINUTES,
        sample_period_minutes=1,
    )
    forecaster = LSTMForecaster(
        feature_cols=LSTM_FEATURE_COLS,
        target_cols=LSTM_TARGET_COLS,
        config=config,
    )
    fit_metrics = forecaster.fit(
        scenario_dfs,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        val_fraction=val_fraction,
    )

    new_metrics = aggregate_lstm_metrics(fit_metrics)
    new_metrics.update({
        "epochs":     epochs,
        "batch_size": batch_size,
        "lookback_minutes": LSTM_LOOKBACK_MINUTES,
        "horizons_minutes": LSTM_HORIZONS_MINUTES,
    })

    try:
        snap = take_snapshot(
            data_dir=data_dir,
            snapshots_dir=data_dir / ".versions",
            message=f"auto: lstm retrain ({epochs} epochs)",
        )
        new_metrics["data_snapshot_id"] = snap.snapshot_id
        new_metrics["data_snapshot_files"] = len(snap.files)
    except Exception as e:
        log.warning("data_snapshot_failed", extra={"error": str(e)})

    promoted = force or should_promote_lstm(new_metrics, baseline, threshold)

    if promoted:
        lstm_dir.mkdir(parents=True, exist_ok=True)
        forecaster.save(lstm_dir)
        save_metrics(new_metrics, metrics_path)
        log.info("lstm_promoted", extra={"path": str(lstm_dir)})
    else:
        log.info("lstm_not_promoted", extra={
            "new_mae_overall":      new_metrics.get("mae_overall"),
            "baseline_mae_overall": baseline.get("mae_overall") if baseline else None,
        })

    return {
        "promoted":         promoted,
        "new_metrics":      new_metrics,
        "baseline_metrics": baseline,
        "model_path":       str(lstm_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Axion AI — Retraining Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["soft_sensor", "lstm", "all"], default="soft_sensor",
        help="Model(s) to retrain",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        metavar="DIR", help="Directory containing scenario CSVs",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Promote new model even if metrics did not improve",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0, metavar="FRACTION",
        help="Minimum relative MAE holdout improvement required for promotion "
             "(0.02 = requires at least 2%% improvement)",
    )
    parser.add_argument(
        "--lstm-epochs", type=int, default=30, metavar="N",
        help="Epochs for LSTM training (only when --model includes lstm)",
    )
    parser.add_argument(
        "--lstm-batch-size", type=int, default=64, metavar="N",
        help="Batch size for LSTM training",
    )
    args = parser.parse_args()

    log.info("retrain_pipeline_start", extra={
        "model": args.model, "force": args.force, "threshold": args.threshold,
    })

    if args.model in ("soft_sensor", "all"):
        with Run("axion-soft-sensor", run_name="retrain") as run:
            run.log_params({
                "pipeline": "retrain",
                "force":     args.force,
                "threshold": args.threshold,
                "data_dir":  str(args.data_dir),
                "n_ensemble": 5,
            })
            result = retrain_soft_sensor(
                data_dir=args.data_dir,
                force=args.force,
                threshold=args.threshold,
            )
            run.log_metrics(result["new_metrics"])
            run.set_tag("promoted", str(result["promoted"]).lower())

        status = "PROMOTED ✓" if result["promoted"] else "SKIPPED (no improvement)"
        new_mae = result["new_metrics"].get("mae_holdout", float("nan"))
        print(f"\nSoft sensor — {status}")
        print(f"  new  MAE holdout: {new_mae:.4f}")
        if result["baseline_metrics"]:
            b_mae = result["baseline_metrics"].get("mae_holdout", float("nan"))
            print(f"  base MAE holdout: {b_mae:.4f}")
        else:
            print("  (no baseline — first run)")

    if args.model in ("lstm", "all"):
        try:
            with Run("axion-lstm-forecaster", run_name="retrain") as lstm_run:
                lstm_run.log_params({
                    "pipeline":   "retrain",
                    "force":      args.force,
                    "threshold":  args.threshold,
                    "data_dir":   str(args.data_dir),
                    "epochs":     args.lstm_epochs,
                    "batch_size": args.lstm_batch_size,
                })
                lstm_result = retrain_lstm(
                    data_dir=args.data_dir,
                    force=args.force,
                    threshold=args.threshold,
                    epochs=args.lstm_epochs,
                    batch_size=args.lstm_batch_size,
                )
                lstm_run.log_metrics(lstm_result["new_metrics"])
                lstm_run.set_tag("promoted", str(lstm_result["promoted"]).lower())

            status = "PROMOTED ✓" if lstm_result["promoted"] else "SKIPPED (no improvement)"
            new_mae = lstm_result["new_metrics"].get("mae_overall", float("nan"))
            print(f"\nLSTM forecaster — {status}")
            print(f"  new  MAE overall: {new_mae:.4f}")
            if lstm_result["baseline_metrics"]:
                b_mae = lstm_result["baseline_metrics"].get("mae_overall", float("nan"))
                print(f"  base MAE overall: {b_mae:.4f}")
            else:
                print("  (no baseline — first run)")
        except RuntimeError as e:
            print(f"\nLSTM retrain skipped: {e}")


if __name__ == "__main__":
    main()
