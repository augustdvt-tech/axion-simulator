"""
Axion AI - Soft Sensor Operational Value Benchmark
==================================================

Measures the operational value of the soft sensor by comparing detection
latency in a realistic scenario: what happens when the primary purity
measurement (cromatography / lab) fails silently, and only the soft sensor
can catch the drift?

Scenario: Take feed_perturbation (normal purity drift from 99% to 92% after
hour 12), and simulate two conditions:

  Condition A: Purity sensor works correctly the whole time
  Condition B: Purity sensor FREEZES at 99.2 starting at hour 10
                (before the event)

For each condition, run the full pipeline twice:
  - Without soft sensor (only R06_PurityDeviation on measured signal)
  - With soft sensor (R06 + R09_SoftSensorDivergence)

Compare:
  - Time to first meaningful recommendation (TTFR)
  - Number of recommendations emitted
  - Whether the operator would have known the product was off-spec

Output: results/soft_sensor_value.png + printed table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analytics import AnalyticalEngine
from recommendations import RecommendationEngine
from soft_sensor import SoftSensor, SoftSensorDetector, PILOT_PURITY_FEATURES

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR  = RESULTS_DIR / "models"


def load_scenario(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def corrupt_purity_sensor(df: pd.DataFrame, freeze_hour: float, freeze_value: float) -> pd.DataFrame:
    """Simulate a frozen purity sensor starting at freeze_hour."""
    df = df.copy()
    t0 = df["timestamp"].min()
    freeze_ts = t0 + pd.Timedelta(hours=freeze_hour)
    mask = df["timestamp"] >= freeze_ts
    df.loc[mask, "column.purity_B"] = freeze_value
    return df


def drift_purity_sensor(df: pd.DataFrame, start_hour: float, drift_per_hour: float = 0.3) -> pd.DataFrame:
    """
    Simulate a slowly-drifting (miscalibrated) purity sensor — retains noise
    so it is NOT detected as frozen, but its absolute value diverges from
    truth. This is a realistic calibration drift mode: 0.3 pct-points/hour
    corresponds to a GC with a slowly contaminating reference standard.
    Adds the drift in the UP direction so the sensor reads artificially
    high — it hides the actual degradation.
    """
    df = df.copy()
    t0 = df["timestamp"].min()
    for i, ts in enumerate(df["timestamp"]):
        hours_after = (ts - t0).total_seconds() / 3600.0
        if hours_after < start_hour:
            continue
        elapsed = hours_after - start_hour
        df.at[i, "column.purity_B"] += drift_per_hour * elapsed
    # Clip to physical range
    df["column.purity_B"] = df["column.purity_B"].clip(upper=100.0)
    return df


def run_pipeline(df: pd.DataFrame, ae: AnalyticalEngine, re: RecommendationEngine) -> list:
    sessions = ae.run_sessions(df)
    return re.generate(sessions, df)


def main():
    print("=" * 72)
    print("Soft Sensor Operational Value Benchmark")
    print("=" * 72)

    # Load pre-trained soft sensor
    sensor = SoftSensor.load(MODELS_DIR / "purity_soft_sensor.joblib")
    ss_detector = SoftSensorDetector(
        sensor=sensor,
        target_tag="column.purity_B",
        abs_tolerance=0.5,
        min_duration_minutes=10.0,
    )

    # Build two engines: baseline and with soft sensor
    df_train = pd.read_csv(DATA_DIR / "normal.csv")
    re_engine = RecommendationEngine()

    def make_engine(with_ss: bool):
        extras = [ss_detector] if with_ss else []
        ae = AnalyticalEngine(
            training_fraction=1.0, warmup_minutes=15.0,
            extra_detectors=extras,
        )
        ae.fit(df_train)
        return ae

    ae_baseline = make_engine(with_ss=False)
    ae_with_ss  = make_engine(with_ss=True)

    # Scenario conditions
    df_healthy = load_scenario("feed_perturbation")
    df_drifted = drift_purity_sensor(df_healthy, start_hour=6.0, drift_per_hour=0.15)

    conditions = [
        ("Healthy sensor, no soft sensor",     df_healthy, ae_baseline),
        ("Healthy sensor, with soft sensor",   df_healthy, ae_with_ss),
        ("Drifting sensor,  no soft sensor",   df_drifted, ae_baseline),
        ("Drifting sensor,  with soft sensor", df_drifted, ae_with_ss),
    ]

    print(f"\n{'condition':<40s}  {'# recs':>7s}  {'TTFR (h)':>10s}  {'rules fired'}")
    print("-" * 100)

    all_results = []
    for label, df, ae in conditions:
        recs = run_pipeline(df, ae, re_engine)
        t0 = df["timestamp"].min()

        # TTFR: time of first actionable (non-LOW) recommendation
        actionable = [r for r in recs if r.urgency.value != "low"]
        if actionable:
            ttfr = (actionable[0].timestamp - t0).total_seconds() / 3600.0
            ttfr_str = f"{ttfr:.2f}"
        else:
            ttfr_str = "—"

        rule_counts = {}
        for r in recs:
            rule_counts[r.rule_fired] = rule_counts.get(r.rule_fired, 0) + 1
        rules_summary = ", ".join(f"{k.replace('_','')[:12]}={v}" for k, v in rule_counts.items())
        all_results.append({
            "label": label, "recs": recs, "df": df, "rules": rule_counts,
            "ttfr": ttfr_str,
        })
        print(f"  {label:<40s}  {len(recs):>7d}  {ttfr_str:>10s}  {rules_summary}")

    # ---- Visualization ----
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    t0 = df_healthy["timestamp"].min()

    for ax, r in zip(axes, all_results):
        df = r["df"]
        recs = r["recs"]
        t_h = (df["timestamp"] - t0).dt.total_seconds() / 3600.0

        # Background: purity signal
        ax.plot(t_h, df["column.purity_B"], color="#bdc3c7", lw=0.8,
                label="Measured purity")

        # If this condition uses soft sensor, overlay the prediction
        if "with soft sensor" in r["label"]:
            preds, stds = sensor.predict_with_confidence(df[sensor.feature_names])
            ax.plot(t_h, preds, color="#c87eff", lw=0.9, alpha=0.8,
                    label="Soft sensor prediction")
            ax.fill_between(t_h, preds - 2*stds, preds + 2*stds,
                            color="#c87eff", alpha=0.12, linewidth=0)

        # Recommendations as scatter at y=97
        rule_colors = {
            "R01_ThermalDrift":         "#e67e22",
            "R02_FeedComposition":      "#c0392b",
            "R03_ControllerOscillation":"#8e44ad",
            "R06_PurityDeviation":      "#2980b9",
            "R09_SoftSensorDivergence": "#c87eff",
        }
        for rec in recs:
            t = (rec.timestamp - t0).total_seconds() / 3600.0
            y = 97.5 if rec.rule_fired == "R09_SoftSensorDivergence" else 97.0
            ax.scatter([t], [y], color=rule_colors.get(rec.rule_fired, "#555"),
                       s=18, alpha=0.7, zorder=3)

        # Event marker at t=12h
        ax.axvline(12.0, color="red", ls="--", lw=0.8, alpha=0.4)
        ax.axhline(98.5, color="red", ls=":",  lw=0.5, alpha=0.3)

        ax.set_title(f"{r['label']}   →   {len(recs)} recs · TTFR(actionable)={r['ttfr']}h",
                     fontsize=10, loc="left")
        ax.set_ylabel("Purity [%]", fontsize=9)
        ax.set_ylim(88, 100.5)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)
        if "Healthy sensor, no soft sensor" in r["label"]:
            ax.legend(loc="lower left", fontsize=8)

    axes[-1].set_xlabel("Scenario time [hours]")
    plt.suptitle("Axion AI — Soft Sensor Operational Value: Detection Under Sensor Failure",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    out_path = RESULTS_DIR / "soft_sensor_value.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")

    # ---- Interpretation ----
    print()
    print("Interpretation:")
    print(f"  A slowly-drifting GC (0.15 pct-points/hour upward bias, starting at hour 6)")
    print(f"  would NOT be caught by the FrozenSensor detector — the sensor still has")
    print(f"  realistic noise. Without the soft sensor, Axion trusts the wrong value.")
    print(f"  With the soft sensor, R09_SoftSensorDivergence fires when the residual")
    print(f"  sustains above tolerance: the operator gets advance warning of a")
    print(f"  miscalibrated cromatograph before QC catches it downstream.")


if __name__ == "__main__":
    main()
