"""
Axion AI - LSTM Forecaster Benchmark Visualization
==================================================

Validates the trained LSTM forecaster on a held-out scenario by sliding the
window across the entire scenario and showing the forecast at fixed
horizons against the actual trajectory.

For each scenario time step (every 30 minutes), forecasts the next 60 min
of the four target variables. Plots the forecast trajectory in light
magenta, with the actual trajectory in dark blue. The visualization
demonstrates: when the model anticipates excursions, when it correctly
captures coupling between variables, and where it loses fidelity at long
horizons.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from predictive import LSTMForecaster

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
LSTM_DIR    = RESULTS_DIR / "models" / "lstm_forecaster"


def main():
    print("Loading trained forecaster...")
    forecaster = LSTMForecaster.load(LSTM_DIR)
    config = forecaster.config

    SCENARIOS = ["feed_perturbation", "thermal_drift",
                 "reactor_instability", "product_grade_change"]

    fig, axes = plt.subplots(
        len(forecaster.target_cols), len(SCENARIOS),
        figsize=(4 * len(SCENARIOS), 2.8 * len(forecaster.target_cols)),
        sharex="col",
    )

    for col_idx, scenario in enumerate(SCENARIOS):
        df = pd.read_csv(DATA_DIR / f"{scenario}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Take a 6h slice to keep figure readable
        df = df.head(360)
        t0 = df["timestamp"].iloc[0]
        t_h = (df["timestamp"] - t0).dt.total_seconds() / 3600.0

        # Slide forecaster every 30 min
        forecast_starts = list(range(config.lookback_steps, len(df) - config.max_horizon_steps, 30))
        forecasts = []
        for end in forecast_starts:
            sub = df.iloc[: end + 1]
            f = forecaster.predict_from_df(sub)
            if f is not None:
                forecasts.append((end, f))

        for tgt_idx, tgt in enumerate(forecaster.target_cols):
            ax = axes[tgt_idx, col_idx] if len(forecaster.target_cols) > 1 else axes[col_idx]

            # Actual
            ax.plot(t_h, df[tgt], color="#1f3b5c", lw=1.0, label="Actual")

            # All forecast trajectories (light magenta)
            for end, f in forecasts:
                start_t = t_h.iloc[end]
                horizon_t = start_t + np.arange(1, config.max_horizon_steps + 1) / 60.0
                ax.plot(horizon_t, f[tgt], color="#c87eff", lw=0.6, alpha=0.5)

            # Highlight one explicit forecast for legend clarity
            if forecasts and tgt_idx == 0 and col_idx == 0:
                end, f = forecasts[len(forecasts) // 2]
                start_t = t_h.iloc[end]
                horizon_t = start_t + np.arange(1, config.max_horizon_steps + 1) / 60.0
                ax.plot(horizon_t, f[tgt], color="#c87eff", lw=1.5,
                        label="LSTM forecast (next 60 min)")

            ax.set_ylabel(tgt.split(".")[-1], fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)
            if tgt_idx == 0:
                ax.set_title(scenario, fontsize=10, loc="left")
            if col_idx == 0 and tgt_idx == 0:
                ax.legend(loc="upper left", fontsize=8)

        if len(forecaster.target_cols) > 1:
            axes[-1, col_idx].set_xlabel("Hours into scenario")

    plt.suptitle(
        "Axion AI — LSTM Forecaster: Sliding-Window Predictions vs Actual",
        fontsize=12, y=1.001,
    )
    plt.tight_layout()
    out_path = RESULTS_DIR / "lstm_forecasts.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
