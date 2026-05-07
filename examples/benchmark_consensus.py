"""
Axion AI - Consensus Benchmark
==============================

End-to-end consensus loop benchmark:

    For each scenario:
      simulator data → analytics → recommendations → consensus → outcomes

For each of three operating modes (advisor, semi-autonomous, autonomous),
runs the full pipeline against all 8 scenarios and measures:

    - Decision distribution (accept / modify / reject / auto)
    - Execution success rate
    - Outcome quality score (predicted vs actual)
    - Per-rule performance

Outputs:
    - results/consensus_summary.csv      cross-scenario summary
    - results/consensus_decisions.csv    full decision log
    - results/consensus_performance.csv  per-rule track record
    - results/consensus_benchmark.png    visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analytics import AnalyticalEngine
from recommendations import RecommendationEngine
from consensus import (
    ConsensusController, OperatingMode, RealisticOperator,
    PerformanceTracker,
    decisions_to_dataframe, outcomes_to_dataframe,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    "normal", "thermal_drift", "feed_perturbation", "reactor_instability",
    "quality_degradation", "energy_waste", "product_grade_change", "sensor_failure",
]


def run_pipeline(scenario, mode, ae_engine, re_engine, operator):
    df = pd.read_csv(DATA_DIR / f"{scenario}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sessions = ae_engine.run_sessions(df)
    recs = re_engine.generate(sessions, df)

    cc = ConsensusController(
        mode=mode, operator=operator,
        # Use a fresh performance tracker per run so cross-scenario stats stay clean
        performance_tracker=PerformanceTracker(),
    )
    log = cc.process(recs, df)
    return df, sessions, recs, log, cc.performance_tracker


def main():
    print("Training analytical engine...")
    df_train = pd.read_csv(DATA_DIR / "normal.csv")
    ae_engine = AnalyticalEngine(training_fraction=1.0, warmup_minutes=15.0)
    ae_engine.fit(df_train)
    re_engine = RecommendationEngine()

    modes = [
        OperatingMode.ADVISOR,
        OperatingMode.SEMI_AUTONOMOUS,
        OperatingMode.AUTONOMOUS_SUPERVISED,
    ]
    operator = RealisticOperator(seed=42)

    summary_rows = []
    all_decisions = []
    all_perf = []
    scenario_data = {}

    for mode in modes:
        print(f"\n=== MODE: {mode.value} ===")
        for scn in SCENARIOS:
            df, sessions, recs, log, perf = run_pipeline(
                scn, mode, ae_engine, re_engine, operator
            )
            s = log.summary()
            scenario_data[(mode.value, scn)] = (df, recs, log, perf)

            # Compute mean quality across outcomes
            quals = [o.quality_score for o in log.outcomes]
            mean_quality = float(np.mean(quals)) if quals else None

            summary_rows.append({
                "mode":              mode.value,
                "scenario":          scn,
                "n_recs":            s["recommendations"],
                "n_accepted":        s["accepted"],
                "n_modified":        s["modified"],
                "n_rejected":        s["rejected"],
                "n_auto":            s["auto_executed"],
                "n_executions":      s["executions"],
                "n_outcomes":        s["outcomes"],
                "mean_quality":      mean_quality,
                "acceptance_rate":   (s["accepted"] + s["modified"] + s["auto_executed"]) / max(1, s["recommendations"]),
            })

            # Append all decisions with mode/scenario tags for the full log
            ddf = decisions_to_dataframe(log.decisions)
            ddf["mode"] = mode.value
            ddf["scenario"] = scn
            all_decisions.append(ddf)

            # Append per-rule performance
            pdf = perf.summary_dataframe()
            if not pdf.empty:
                pdf["mode"] = mode.value
                pdf["scenario"] = scn
                all_perf.append(pdf)

    summary = pd.DataFrame(summary_rows)
    full_decisions = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    full_perf = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()

    summary.to_csv(RESULTS_DIR / "consensus_summary.csv", index=False)
    full_decisions.to_csv(RESULTS_DIR / "consensus_decisions.csv", index=False)
    full_perf.to_csv(RESULTS_DIR / "consensus_performance.csv", index=False)

    # ---- Print summary tables ----
    print("\n\n" + "=" * 130)
    print("CONSENSUS BENCHMARK SUMMARY")
    print("=" * 130)
    for mode in modes:
        print(f"\n{mode.value.upper():>22s}")
        sub = summary[summary["mode"] == mode.value]
        print(f"  {'scenario':<25} {'recs':>5} {'accept':>7} {'modify':>7} "
              f"{'reject':>7} {'auto':>5} {'exec':>5} {'outcm':>6} {'qual':>6}")
        print("  " + "-" * 100)
        for _, r in sub.iterrows():
            q_str = f"{r['mean_quality']:.0%}" if r["mean_quality"] is not None else "—"
            print(f"  {r['scenario']:<25} {r['n_recs']:>5d} {r['n_accepted']:>7d} "
                  f"{r['n_modified']:>7d} {r['n_rejected']:>7d} {r['n_auto']:>5d} "
                  f"{r['n_executions']:>5d} {r['n_outcomes']:>6d} {q_str:>6}")
        # Totals row
        tot = sub.sum(numeric_only=True)
        print(f"  {'TOTAL':<25} {int(tot['n_recs']):>5d} {int(tot['n_accepted']):>7d} "
              f"{int(tot['n_modified']):>7d} {int(tot['n_rejected']):>7d} "
              f"{int(tot['n_auto']):>5d} {int(tot['n_executions']):>5d} "
              f"{int(tot['n_outcomes']):>6d}")

    # ---- Per-rule track record (semi_autonomous only) ----
    print("\n" + "=" * 130)
    print("PER-RULE TRACK RECORD (semi-autonomous mode, aggregated across all scenarios)")
    print("=" * 130)
    semi_perf = full_perf[full_perf["mode"] == "semi_autonomous"]
    if not semi_perf.empty:
        agg = semi_perf.groupby("rule").agg({
            "total_recs": "sum",
            "accepted": "sum",
            "modified": "sum",
            "rejected": "sum",
            "outcomes_measured": "sum",
            "successes": "sum",
            "failures": "sum",
        }).reset_index()
        agg["acceptance_rate"] = (agg["accepted"] + agg["modified"]) / agg["total_recs"]
        agg["success_rate"] = agg["successes"] / agg["outcomes_measured"].replace(0, np.nan)
        agg = agg.sort_values("total_recs", ascending=False)
        print(f"  {'rule':<28} {'recs':>5} {'acc':>5} {'mod':>5} {'rej':>5} "
              f"{'meas':>5} {'succ':>5} {'fail':>5} {'acc%':>6} {'succ%':>6}")
        print("  " + "-" * 100)
        for _, r in agg.iterrows():
            ar = f"{r['acceptance_rate']:.0%}"
            sr = f"{r['success_rate']:.0%}" if not pd.isna(r["success_rate"]) else "—"
            print(f"  {r['rule']:<28} {int(r['total_recs']):>5d} {int(r['accepted']):>5d} "
                  f"{int(r['modified']):>5d} {int(r['rejected']):>5d} "
                  f"{int(r['outcomes_measured']):>5d} {int(r['successes']):>5d} "
                  f"{int(r['failures']):>5d} {ar:>6} {sr:>6}")

    # ---- Decision detail samples ----
    print("\n" + "=" * 130)
    print("SAMPLE DECISION FLOW — feed_perturbation, semi_autonomous mode")
    print("=" * 130)
    df, recs, log, _ = scenario_data[("semi_autonomous", "feed_perturbation")]
    for i, (rec, dec) in enumerate(zip(recs[:5], log.decisions[:5])):
        print(f"\n  [{i+1}] REC {rec.id}: {rec.diagnosis[:75]}")
        print(f"      → {dec.status.value.upper()}: {dec.justification[:80]}")
        if dec.actual_action and dec.actual_action.is_automated:
            print(f"      action: {dec.actual_action.target_variable} → "
                  f"{dec.actual_action.proposed_value:.3f} "
                  f"(Δ={dec.actual_action.adjustment:+.3f})")

    # ---- Visualization: 3 modes × outcome quality ----
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    mode_names = ["advisor", "semi_autonomous", "autonomous_supervised"]
    mode_titles = ["ADVISOR (operator manual)", "SEMI-AUTONOMOUS (one-click approval)",
                   "AUTONOMOUS SUPERVISED (auto within rules)"]
    bar_width = 0.18

    for ax, mode_name, title in zip(axes, mode_names, mode_titles):
        sub = summary[summary["mode"] == mode_name]
        x = np.arange(len(sub))
        ax.bar(x - 1.5 * bar_width, sub["n_accepted"], bar_width,
               label="Accepted", color="#27ae60")
        ax.bar(x - 0.5 * bar_width, sub["n_modified"], bar_width,
               label="Modified", color="#f39c12")
        ax.bar(x + 0.5 * bar_width, sub["n_rejected"], bar_width,
               label="Rejected", color="#e74c3c")
        ax.bar(x + 1.5 * bar_width, sub["n_auto"], bar_width,
               label="Auto-executed", color="#3498db")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["scenario"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("# Decisions", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        if mode_name == "advisor":
            ax.legend(loc="upper right", fontsize=8, ncol=4)

        # Annotate quality if available
        for i, (_, r) in enumerate(sub.iterrows()):
            if r["mean_quality"] is not None:
                ax.text(i, ax.get_ylim()[1] * 0.95, f"q={r['mean_quality']:.0%}",
                        ha="center", fontsize=7, color="#555")

    plt.suptitle("Axion AI — Consensus Loop Benchmark Across Operating Modes",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    out_path = RESULTS_DIR / "consensus_benchmark.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    print(f"Summaries saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
