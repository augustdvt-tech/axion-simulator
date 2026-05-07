"""
Axion AI - Recommendations Benchmark
====================================

Full end-to-end pipeline:
    CSV data → AnalyticalEngine → event sessions → RecommendationEngine
                                                → list of Recommendations

For each of the 8 scenarios this produces:
    1. A summary: how many recommendations by rule
    2. Time-to-first-recommendation after each known event
    3. Full recommendation cards for manual review
    4. A figure showing the recommendation timeline with severity coloring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analytics import AnalyticalEngine
from recommendations import RecommendationEngine, recommendations_to_dataframe

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCENARIO_GROUND_TRUTH = {
    "normal":               {"event_time_h": None, "expected_rules": []},
    "thermal_drift":        {"event_time_h": 0.0,  "expected_rules": ["R01_ThermalDrift"]},
    "feed_perturbation":    {"event_time_h": 12.0, "expected_rules": ["R02_FeedComposition", "R06_PurityDeviation"]},
    "reactor_instability":  {"event_time_h": 6.0,  "expected_rules": ["R03_ControllerOscillation"]},
    "quality_degradation":  {"event_time_h": 0.0,  "expected_rules": ["R04_ColumnEfficiencyLoss", "R06_PurityDeviation"]},
    "energy_waste":         {"event_time_h": 4.0,  "expected_rules": ["R05_ExcessReflux"]},
    "product_grade_change": {"event_time_h": 10.0, "expected_rules": ["R08_ProductTransition"]},
    "sensor_failure":       {"event_time_h": 8.0,  "expected_rules": ["R07_SensorFault"]},
}

URGENCY_COLOR = {
    "critical": "#c0392b",
    "high":     "#e67e22",
    "medium":   "#f39c12",
    "low":      "#27ae60",
}


def main():
    print("Training analytical engine on normal.csv...")
    df_train = pd.read_csv(DATA_DIR / "normal.csv")
    ae = AnalyticalEngine(
        training_fraction=1.0,
        warmup_minutes=15.0,
        session_gap_minutes=30.0,
    )
    ae.fit(df_train)
    re = RecommendationEngine()

    print(f"Loaded {len(re.rules)} diagnostic rules.\n")

    scenario_data = {}
    summary_rows = []

    for scn, gt in SCENARIO_GROUND_TRUTH.items():
        print(f"  Evaluating: {scn}")
        df = pd.read_csv(DATA_DIR / f"{scn}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        sessions = ae.run_sessions(df)
        recs = re.generate(sessions, df)
        recs_df = recommendations_to_dataframe(recs)
        scenario_data[scn] = (df, sessions, recs, recs_df)

        # Count recommendations by rule
        rule_counts = recs_df["rule_fired"].value_counts().to_dict() if not recs_df.empty else {}

        # Time-to-first-recommendation per expected rule
        ttfr = {}
        t0 = df["timestamp"].min()
        if gt["event_time_h"] is not None and not recs_df.empty:
            event_ts = t0 + pd.Timedelta(hours=gt["event_time_h"])
            for rule in gt["expected_rules"]:
                matching = recs_df[recs_df["rule_fired"] == rule]
                if not matching.empty:
                    # First recommendation for this rule (regardless of whether
                    # it fires before or after the nominal event time — event
                    # start times are noisy approximations)
                    first_ts = matching.iloc[0]["timestamp"]
                    ttfr[rule] = (first_ts - event_ts).total_seconds() / 60.0

        summary_rows.append({
            "scenario":         scn,
            "total_sessions":   len(sessions),
            "total_recs":       len(recs),
            "n_critical":       int((recs_df["urgency"] == "critical").sum()) if not recs_df.empty else 0,
            "n_high":           int((recs_df["urgency"] == "high").sum()) if not recs_df.empty else 0,
            "n_medium":         int((recs_df["urgency"] == "medium").sum()) if not recs_df.empty else 0,
            "n_low":            int((recs_df["urgency"] == "low").sum()) if not recs_df.empty else 0,
            "expected_rules":   ", ".join(gt["expected_rules"]) if gt["expected_rules"] else "—",
            "rules_detected":   ", ".join(sorted(set(rule_counts.keys()))) if rule_counts else "—",
            "ttfr_minutes":     "; ".join(f"{k}:{v:.0f}" for k, v in ttfr.items()) if ttfr else "—",
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "benchmark_recommendations.csv", index=False)

    # ---- Print summary ----
    print("\n" + "=" * 120)
    print("RECOMMENDATION BENCHMARK SUMMARY")
    print("=" * 120)
    print(f"{'scenario':<25} {'sessions':>9} {'recs':>5} {'crit':>5} {'high':>5} {'med':>5} {'low':>5}  {'rules detected'}")
    print("-" * 120)
    for _, r in summary.iterrows():
        rules = r["rules_detected"][:55]
        print(f"{r['scenario']:<25} {r['total_sessions']:>9d} {r['total_recs']:>5d} "
              f"{r['n_critical']:>5d} {r['n_high']:>5d} {r['n_medium']:>5d} {r['n_low']:>5d}  "
              f"{rules}")

    print("\n" + "=" * 120)
    print("TIME-TO-FIRST-RECOMMENDATION after known event (minutes)")
    print("=" * 120)
    for _, r in summary.iterrows():
        expected = r["expected_rules"]
        ttfr = r["ttfr_minutes"]
        print(f"  {r['scenario']:<25s}")
        print(f"     expected: {expected}")
        print(f"     detected: {ttfr}")

    # ---- Print sample cards from interesting scenarios ----
    for scn in ["thermal_drift", "feed_perturbation", "energy_waste",
                "reactor_instability", "sensor_failure"]:
        _, _, recs, _ = scenario_data[scn]
        print("\n" + "=" * 120)
        print(f"SCENARIO: {scn} — first recommendation of each rule")
        print("=" * 120)
        seen = set()
        for r in recs:
            if r.rule_fired in seen:
                continue
            seen.add(r.rule_fired)
            print(r.format_detail())
            print()

    # ---- Plot: Recommendation timeline ----
    n_scn = len(SCENARIO_GROUND_TRUTH)
    fig, axes = plt.subplots(n_scn, 1, figsize=(14, 1.8 * n_scn))

    # List all rules we'll ever plot (for consistent y-axis)
    all_rules = sorted(set(
        rec.rule_fired for _, _, recs, _ in scenario_data.values() for rec in recs
    ))
    if not all_rules:
        all_rules = ["(no recommendations)"]
    rule_y = {r: i for i, r in enumerate(all_rules)}

    for ax, (scn, gt) in zip(axes, SCENARIO_GROUND_TRUTH.items()):
        df, _, recs, recs_df = scenario_data[scn]
        t0 = df["timestamp"].min()
        t_h = (df["timestamp"] - t0).dt.total_seconds() / 3600

        # Background (purity)
        ax2 = ax.twinx()
        ax2.plot(t_h, df["column.purity_B"], color="#bdc3c7", lw=0.7)
        ax2.axhline(98.5, color="red", ls=":", lw=0.8, alpha=0.4)
        ax2.set_ylabel("Purity [%]", color="#7f8c8d", fontsize=8)
        ax2.tick_params(axis='y', labelsize=7, colors="#7f8c8d")

        # Plot each recommendation as a colored dot
        if recs:
            for rec in recs:
                if rec.rule_fired not in rule_y:
                    continue
                x = (rec.timestamp - t0).total_seconds() / 3600
                y = rule_y[rec.rule_fired]
                color = URGENCY_COLOR.get(rec.urgency.value, "#7f8c8d")
                ax.scatter([x], [y], color=color, s=60, alpha=0.85,
                           edgecolors="black", linewidths=0.4, zorder=3)

        if gt["event_time_h"] is not None:
            ax.axvline(gt["event_time_h"], color="red", ls="--", lw=1.2, alpha=0.5)
            ax.text(gt["event_time_h"], len(all_rules) - 0.3,
                    " event", color="red", fontsize=7, ha="left", va="bottom")

        ax.set_yticks(list(rule_y.values()))
        ax.set_yticklabels(list(rule_y.keys()), fontsize=7)
        ax.set_xlim(0, t_h.max())
        ax.set_ylim(-0.7, len(all_rules) - 0.3)
        ax.set_title(f"{scn} ({len(recs)} recommendations)",
                     fontsize=10, loc="left")
        ax.grid(True, alpha=0.2, axis="x")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=u.upper())
        for u, c in URGENCY_COLOR.items()
    ]
    axes[0].legend(
        handles=legend_patches, loc="upper right",
        fontsize=8, title="Urgency", title_fontsize=8, framealpha=0.9,
    )

    axes[-1].set_xlabel("Time [hours]")
    plt.suptitle("Axion AI — Recommendation Timeline by Scenario",
                 fontsize=13, y=1.001)
    plt.tight_layout()
    out_path = RESULTS_DIR / "recommendations_benchmark.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    print(f"Summary saved to: {RESULTS_DIR / 'benchmark_recommendations.csv'}")


if __name__ == "__main__":
    main()
