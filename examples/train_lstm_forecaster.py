"""
Axion AI - LSTM Forecaster Training
====================================

Trains the multi-horizon LSTM forecaster on combined scenario data.

Training strategy
-----------------
Combine 7 scenarios (excluding sensor_failure to avoid corrupted ground
truth). Build sliding windows independently per scenario — windows do NOT
cross scenario boundaries, since the dynamics in different scenarios are
discontinuous. Time-split the combined window pool into train / val.

Targets and features
--------------------
Targets (what we predict at horizons +5, +15, +30, +60 min):
  - cstr.T_R_C       — reactor temperature (control critical)
  - column.purity_B  — product purity (quality critical)
  - column.Q_reb_kW  — energy consumption (cost critical)
  - cstr.conversion  — reactor performance

Features (what the model sees as context):
  - All targets above (temporal lags help)
  - Manipulated variables: column.RR, cstr.F_cool, cstr.F_feed
  - Disturbances: cstr.C_A, cstr.T_feed_C
  - Auxiliary measurements: column.T_bot_C, column.T_top_C, cstr.T_J_C

This is the classic "exogenous + endogenous" feature set: the model knows
both the current operating state and the recent behavior of the targets.

Output
------
results/models/lstm_forecaster/      Keras model + scalers + metrics
results/lstm_training_history.png    Loss curves
results/lstm_horizon_metrics.png     MAE/R² per horizon per target
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from predictive import (
    WindowConfig, LSTMForecaster, LSTMMetrics,
)
from axion_mlflow import Run

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR  = RESULTS_DIR / "models"
LSTM_DIR    = MODELS_DIR / "lstm_forecaster"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_SCENARIOS = [
    "normal", "thermal_drift", "feed_perturbation",
    "reactor_instability", "quality_degradation",
    "energy_waste", "product_grade_change",
]

FEATURE_COLS = [
    # Endogenous (targets themselves help)
    "cstr.T_R_C", "column.purity_B", "column.Q_reb_kW", "cstr.conversion",
    # Manipulated
    "column.RR", "cstr.F_cool", "cstr.F_feed",
    # Disturbances
    "cstr.C_A", "cstr.T_feed_C",
    # Auxiliary
    "column.T_bot_C", "column.T_top_C", "cstr.T_J_C",
]

TARGET_COLS = [
    "cstr.T_R_C",
    "column.purity_B",
    "column.Q_reb_kW",
    "cstr.conversion",
]


def load_scenarios() -> list[pd.DataFrame]:
    out = []
    for s in TRAIN_SCENARIOS:
        df = pd.read_csv(DATA_DIR / f"{s}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        out.append(df)
    return out


def make_training_history_figure(metrics: LSTMMetrics, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(metrics.train_loss_history) + 1)
    ax.plot(epochs, metrics.train_loss_history, "-o",
            color="#2980b9", lw=1.5, ms=4, label="Train loss (MSE)")
    ax.plot(epochs, metrics.val_loss_history, "-s",
            color="#e74c3c", lw=1.5, ms=4, label="Validation loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized space)")
    ax.set_title("LSTM Forecaster — Training Convergence", fontsize=12, loc="left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"   Figure saved: {out_path}")
    plt.close()


def make_horizon_metrics_figure(metrics: LSTMMetrics, out_path: Path) -> None:
    horizons = sorted(metrics.by_horizon.keys())
    targets = list(metrics.by_horizon[horizons[0]].keys()) if horizons else []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # MAE per horizon per target
    ax = axes[0]
    for tgt in targets:
        maes = [metrics.by_horizon[h][tgt]["mae"] for h in horizons]
        ax.plot(horizons, maes, "-o", lw=1.5, ms=5, label=tgt.split(".")[-1])
    ax.set_xlabel("Forecast horizon [minutes]")
    ax.set_ylabel("MAE (original units)")
    ax.set_title("Forecast accuracy degrades with horizon", fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # R² per horizon per target
    ax = axes[1]
    for tgt in targets:
        r2s = [metrics.by_horizon[h][tgt]["r2"] for h in horizons]
        ax.plot(horizons, r2s, "-o", lw=1.5, ms=5, label=tgt.split(".")[-1])
    ax.axhline(0, color="red", ls=":", lw=0.5, alpha=0.4)
    ax.set_xlabel("Forecast horizon [minutes]")
    ax.set_ylabel("R²")
    ax.set_title("Variance explained by horizon and target", fontsize=11, loc="left")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle("Axion AI — LSTM Forecaster: Per-Horizon Metrics on Validation Set",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"   Figure saved: {out_path}")
    plt.close()


def main():
    print("=" * 72)
    print("Axion AI — LSTM Forecaster Training")
    print("=" * 72)

    print("\n1) Loading scenarios...")
    dfs = load_scenarios()
    total_rows = sum(len(df) for df in dfs)
    print(f"   {len(dfs)} scenarios, {total_rows} samples total")

    config = WindowConfig(
        lookback_minutes=45,
        horizons_minutes=[5, 15, 30, 60],
        sample_period_minutes=1.0,
    )
    units1, units2, dropout = 48, 24, 0.15
    epochs, batch_size = 20, 128

    with Run("axion-lstm-forecaster", run_name="lstm_v1") as run:
        run.log_params({
            "lookback_minutes": config.lookback_minutes,
            "horizons_minutes": str(config.horizons_minutes),
            "n_features":       len(FEATURE_COLS),
            "n_targets":        len(TARGET_COLS),
            "units1":           units1,
            "units2":           units2,
            "dropout":          dropout,
            "epochs":           epochs,
            "batch_size":       batch_size,
            "train_scenarios":  len(TRAIN_SCENARIOS),
            "total_samples":    total_rows,
        })

        print("\n2) Building forecaster...")
        print(f"   Lookback: {config.lookback_minutes} min  "
              f"({config.lookback_steps} steps)")
        print(f"   Horizons: {config.horizons_minutes} min")
        print(f"   Features: {len(FEATURE_COLS)}  Targets: {len(TARGET_COLS)}")

        forecaster = LSTMForecaster(
            feature_cols=FEATURE_COLS,
            target_cols=TARGET_COLS,
            config=config,
            units1=units1, units2=units2, dropout=dropout,
        )

        print("\n3) Training...")
        metrics = forecaster.fit(
            dfs, epochs=epochs, batch_size=batch_size, verbose=2, val_fraction=0.2,
        )
        print()
        print("   Training metrics summary:")
        print(metrics.format())

        run.log_epoch_metrics(metrics.train_loss_history, metrics.val_loss_history)

        for horizon, tgt_metrics in metrics.by_horizon.items():
            for tgt, vals in tgt_metrics.items():
                short = tgt.split(".")[-1]
                run.log_metrics({
                    f"mae_{short}_{horizon}min": vals["mae"],
                    f"r2_{short}_{horizon}min":  vals["r2"],
                })

        print("\n4) Saving model...")
        forecaster.save(LSTM_DIR)
        print(f"   Model saved: {LSTM_DIR}")

        print("\n5) Generating figures...")
        hist_path = RESULTS_DIR / "lstm_training_history.png"
        horiz_path = RESULTS_DIR / "lstm_horizon_metrics.png"
        make_training_history_figure(metrics, hist_path)
        make_horizon_metrics_figure(metrics, horiz_path)

        run.log_artifacts(str(LSTM_DIR), artifact_path="model")
        run.log_artifact(str(hist_path))
        run.log_artifact(str(horiz_path))

    print("\n" + "=" * 72)
    print("Training complete.")


if __name__ == "__main__":
    main()
