"""
Axion AI - Soft Sensor Training
===============================

Trains the purity soft sensor using the simulator scenarios as ground truth.

Strategy
--------
The training pipeline assembles a combined dataset from all 8 scenarios
EXCEPT `sensor_failure` (where the sensor itself is frozen and would
contaminate the training distribution). The combined dataset spans the
full operating envelope Axion AI is expected to handle.

We then:
  1. Fit the PuritySoftSensor on the combined dataset (80/20 split).
  2. Evaluate per-scenario to see how the sensor performs in each regime.
  3. Serialize the trained model to results/purity_soft_sensor.joblib
     for use by the API/UI in live inference.

The per-scenario evaluation is what tells a process engineer whether the
sensor is trustworthy in the scenarios they care about: nominal operation,
feed disturbances, product transitions, etc. A good soft sensor should
have:
  - Low MAE across all scenarios
  - No systematic bias (residuals centered on zero)
  - Uncertainty that grows in transient / transition periods
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from soft_sensor import PuritySoftSensor, PILOT_PURITY_FEATURES, TARGET_PURITY
from axion_mlflow import Run

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR  = Path(__file__).parent.parent / "results" / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Training scenarios: all except sensor_failure (contaminated ground truth)
TRAIN_SCENARIOS = [
    "normal", "thermal_drift", "feed_perturbation",
    "reactor_instability", "quality_degradation",
    "energy_waste", "product_grade_change",
]

# Scenario we hold out entirely for realistic per-scenario evaluation
HOLDOUT_SCENARIOS = ["sensor_failure"]


def load_combined() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble combined dataset from training scenarios with a scenario tag."""
    frames = []
    for s in TRAIN_SCENARIOS:
        df = pd.read_csv(DATA_DIR / f"{s}.csv")
        df["_scenario"] = s
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    # Require all features + target to be present
    needed = PILOT_PURITY_FEATURES + [TARGET_PURITY]
    mask = combined[needed].notna().all(axis=1)
    combined = combined.loc[mask].reset_index(drop=True)

    X = combined[PILOT_PURITY_FEATURES].copy()
    y = combined[TARGET_PURITY].copy()
    scn = combined["_scenario"].copy()
    return X, y, scn


def per_scenario_metrics(sensor: PuritySoftSensor,
                         X: pd.DataFrame, y: pd.Series, scn: pd.Series) -> pd.DataFrame:
    """Evaluate the trained sensor per scenario."""
    preds, stds = sensor.predict_with_confidence(X)
    out = pd.DataFrame({
        "scenario":  scn.values,
        "actual":    y.values,
        "predicted": preds,
        "std":       stds,
    })
    out["residual"] = out["actual"] - out["predicted"]

    rows = []
    for s in TRAIN_SCENARIOS:
        sub = out[out["scenario"] == s]
        if sub.empty:
            continue
        rows.append({
            "scenario":     s,
            "n_samples":    len(sub),
            "mae":          float(np.mean(np.abs(sub["residual"]))),
            "rmse":         float(np.sqrt(np.mean(sub["residual"] ** 2))),
            "bias":         float(np.mean(sub["residual"])),
            "max_abs_err":  float(np.max(np.abs(sub["residual"]))),
            "mean_std":     float(np.mean(sub["std"])),
            "p95_std":      float(np.percentile(sub["std"], 95)),
        })
    return pd.DataFrame(rows)


def make_figure(sensor, X, y, scn, out_path):
    """Plot per-scenario predicted vs actual over time."""
    preds, stds = sensor.predict_with_confidence(X)
    fig, axes = plt.subplots(
        len(TRAIN_SCENARIOS), 1, figsize=(13, 1.7 * len(TRAIN_SCENARIOS)),
        sharex=False,
    )

    for ax, s in zip(axes, TRAIN_SCENARIOS):
        idx = scn.values == s
        if not idx.any():
            continue
        t = np.arange(idx.sum()) / 60.0   # samples are 1/min → x in hours
        actual = y.values[idx]
        pred   = preds[idx]
        sd     = stds[idx]

        # Uncertainty band
        ax.fill_between(t, pred - 2*sd, pred + 2*sd, color="#3498db",
                        alpha=0.15, label="±2σ", linewidth=0)
        ax.plot(t, actual, color="#2c3e50", lw=0.8, label="Actual")
        ax.plot(t, pred,   color="#e74c3c", lw=0.8, label="Soft sensor")
        ax.axhline(98.5, color="#e74c3c", ls=":", lw=0.5, alpha=0.4)

        mae = float(np.mean(np.abs(actual - pred)))
        ax.set_title(f"{s}   (MAE = {mae:.3f})", fontsize=10, loc="left")
        ax.set_ylabel("Purity [%]", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)
        if s == TRAIN_SCENARIOS[0]:
            ax.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("Scenario time [hours]")
    plt.suptitle("Axion AI — Purity Soft Sensor: Predicted vs Actual",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"  Figure saved: {out_path}")


def main():
    print("=" * 72)
    print("Axion AI — Purity Soft Sensor Training")
    print("=" * 72)

    print("\n1) Loading combined training dataset...")
    X, y, scn = load_combined()
    print(f"   Samples: {len(X)} across {len(TRAIN_SCENARIOS)} scenarios")
    print(f"   Features: {list(X.columns)}")
    print(f"   Target: {TARGET_PURITY}  range=[{y.min():.2f}, {y.max():.2f}]  "
          f"mean={y.mean():.3f}  std={y.std():.3f}")

    with Run("axion-soft-sensor", run_name="gbr_v1") as run:
        run.log_params({
            "n_ensemble":       5,
            "n_estimators":     200,
            "n_features":       len(PILOT_PURITY_FEATURES),
            "train_scenarios":  len(TRAIN_SCENARIOS),
            "n_samples":        len(X),
        })

        print("\n2) Fitting soft sensor (ensemble of gradient-boosted trees)...")
        sensor = PuritySoftSensor(n_ensemble=5)
        metrics = sensor.fit(X, y)
        print(f"   Train/test split metrics:")
        print(f"     {metrics.format()}")

        run.log_metrics({"mae": metrics.mae, "rmse": metrics.rmse, "r2": metrics.r2})

        print("\n3) Feature importances:")
        for _, row in sensor.feature_importances().iterrows():
            print(f"     {row['feature']:25s}  {row['importance']:.3f}")

        print("\n4) Per-scenario evaluation:")
        per_scn = per_scenario_metrics(sensor, X, y, scn)
        print(f"   {'scenario':<23s}  {'n':>5s}  {'MAE':>7s}  {'RMSE':>7s}  "
              f"{'bias':>7s}  {'max|err|':>8s}  {'mean_σ':>7s}")
        print("   " + "-" * 72)
        for _, r in per_scn.iterrows():
            print(f"   {r['scenario']:<23s}  {int(r['n_samples']):>5d}  "
                  f"{r['mae']:>7.3f}  {r['rmse']:>7.3f}  "
                  f"{r['bias']:>+7.3f}  {r['max_abs_err']:>8.3f}  "
                  f"{r['mean_std']:>7.3f}")
            run.log_metric(f"mae_{r['scenario']}", r["mae"])

        csv_path = RESULTS_DIR / "soft_sensor_per_scenario.csv"
        per_scn.to_csv(csv_path, index=False)

        print("\n5) Saving model + generating figure...")
        model_path = MODELS_DIR / "purity_soft_sensor.joblib"
        sensor.save(model_path)
        print(f"   Model saved: {model_path}")

        fig_path = RESULTS_DIR / "soft_sensor_predictions.png"
        make_figure(sensor, X, y, scn, fig_path)

        run.log_artifact(str(model_path))
        run.log_artifact(str(csv_path))
        run.log_artifact(str(fig_path))

        # ---- Validation on a held-out scenario ----
        print("\n6) Held-out scenario evaluation:")
        for s in HOLDOUT_SCENARIOS:
            df_h = pd.read_csv(DATA_DIR / f"{s}.csv")
            mask = df_h[PILOT_PURITY_FEATURES + [TARGET_PURITY]].notna().all(axis=1)
            if not mask.any():
                continue
            Xh = df_h.loc[mask, PILOT_PURITY_FEATURES]
            yh = df_h.loc[mask, TARGET_PURITY]
            preds, stds = sensor.predict_with_confidence(Xh)
            mae = float(np.mean(np.abs(yh.values - preds)))
            bias = float(np.mean(yh.values - preds))
            print(f"   {s}: n={len(Xh)}  MAE={mae:.3f}  bias={bias:+.3f}")
            run.log_metric(f"mae_holdout_{s}", mae)

    print("\n" + "=" * 72)
    print("Training complete.")


if __name__ == "__main__":
    main()
