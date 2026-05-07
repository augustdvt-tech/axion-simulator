"""
Axion AI - Analytics Benchmark (v2: event sessions)
===================================================

Runs the AnalyticalEngine against all 8 scenarios using `normal.csv` as the
training baseline, and produces:

1. A raw-alerts summary (diagnostic, not operator-facing)
2. An EVENT SESSIONS summary — what an operator would actually see on screen
3. A multi-panel figure showing the session timeline

Event sessions group hundreds of raw alerts into a single "event", massively
reducing alert fatigue while preserving the information content.

Configuration:
- Train on `normal.csv` (full file)
- 15 min warmup window applied to every evaluation (filters startup transient)
- Sessions grouped with a 30 min gap
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analytics import AnalyticalEngine, alerts_to_dataframe, sessions_to_dataframe

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SCENARIO_GROUND_TRUTH = {
    "normal":               {"event_time_h": None, "description": "No event — baseline"},
    "thermal_drift":        {"event_time_h": 0.0,  "description": "Jacket fouling drift starts at t=0"},
    "feed_perturbation":    {"event_time_h": 12.0, "description": "+15% feed concentration step at t=12h"},
    "reactor_instability":  {"event_time_h": 6.0,  "description": "Sinusoidal coolant instability at t=6h"},
    "quality_degradation":  {"event_time_h": 0.0,  "description": "Volatility α decay starts at t=0"},
    "energy_waste":         {"event_time_h": 4.0,  "description": "Reflux step +1.5 at t=4h"},
    "product_grade_change": {"event_time_h": 10.0, "description": "RR setpoint 5.5→4.0 at t=10h"},
    "sensor_failure":       {"event_time_h": 8.0,  "description": "T_R sensor frozen 8h–12h"},
}


def time_to_first_session(sessions_df, detector, event_time_h, t0):
    if sessions_df.empty or event_time_h is None:
        return np.nan
    sub = sessions_df[sessions_df["detector"] == detector]
    if sub.empty:
        return np.nan
    event_ts = t0 + pd.Timedelta(hours=event_time_h)
    after = sub[sub["start_time"] >= event_ts]
    if after.empty:
        return np.nan
    return (after.iloc[0]["start_time"] - event_ts).total_seconds() / 60.0


def main():
    print("Training analytical engine on normal.csv (24 h baseline)...")
    df_train = pd.read_csv(DATA_DIR / "normal.csv")
    engine = AnalyticalEngine(
        training_fraction=1.0,
        warmup_minutes=15.0,
        session_gap_minutes=30.0,
    )
    engine.fit(df_train)
    print(f"  PCA model: {engine.pca.model.n_components} components, "
          f"{engine.pca.model.variance_explained:.1%} variance explained")
    print(f"  T² limit = {engine.pca.model.t2_limit:.3f}, "
          f"SPE limit = {engine.pca.model.spe_limit:.3f}\n")

    raw_rows = []
    session_rows = []
    scenario_results = {}

    for scn, gt in SCENARIO_GROUND_TRUTH.items():
        print(f"  Evaluating: {scn}")
        df = pd.read_csv(DATA_DIR / f"{scn}.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        raw_alerts = engine.run(df)
        sessions = engine.run_sessions(df)
        raw_df = alerts_to_dataframe(raw_alerts)
        sess_df = sessions_to_dataframe(sessions)
        scenario_results[scn] = (df, raw_df, sess_df)

        t0 = df["timestamp"].min()
        raw_rows.append({
            "scenario":     scn,
            "total_alerts": len(raw_df),
            "shewhart":     int((raw_df["detector"] == "SPC.Shewhart").sum()),
            "ewma":         int((raw_df["detector"] == "SPC.EWMA").sum()),
            "pca_t2":       int((raw_df["detector"] == "PCA.T2").sum()),
            "pca_spe":      int((raw_df["detector"] == "PCA.SPE").sum()),
            "trend":        int((raw_df["detector"] == "Trend.Projection").sum()),
            "regime":       int((raw_df["detector"] == "Regime.CUSUM").sum()),
        })
        session_rows.append({
            "scenario":          scn,
            "event_time_h":      gt["event_time_h"],
            "total_sessions":    len(sess_df),
            "shewhart":          int((sess_df["detector"] == "SPC.Shewhart").sum()),
            "ewma":              int((sess_df["detector"] == "SPC.EWMA").sum()),
            "pca_t2":            int((sess_df["detector"] == "PCA.T2").sum()),
            "pca_spe":           int((sess_df["detector"] == "PCA.SPE").sum()),
            "trend":             int((sess_df["detector"] == "Trend.Projection").sum()),
            "regime":            int((sess_df["detector"] == "Regime.CUSUM").sum()),
            "ttfs_shewhart_min": time_to_first_session(sess_df, "SPC.Shewhart", gt["event_time_h"], t0),
            "ttfs_ewma_min":     time_to_first_session(sess_df, "SPC.EWMA", gt["event_time_h"], t0),
            "ttfs_pca_t2_min":   time_to_first_session(sess_df, "PCA.T2", gt["event_time_h"], t0),
            "ttfs_pca_spe_min":  time_to_first_session(sess_df, "PCA.SPE", gt["event_time_h"], t0),
            "ttfs_trend_min":    time_to_first_session(sess_df, "Trend.Projection", gt["event_time_h"], t0),
            "ttfs_regime_min":   time_to_first_session(sess_df, "Regime.CUSUM", gt["event_time_h"], t0),
        })

    raw_summary = pd.DataFrame(raw_rows)
    sess_summary = pd.DataFrame(session_rows)
    raw_summary.to_csv(RESULTS_DIR / "benchmark_raw_alerts.csv", index=False)
    sess_summary.to_csv(RESULTS_DIR / "benchmark_sessions.csv", index=False)

    print("\n" + "=" * 100)
    print("RAW ALERTS (high volume — internal diagnostic only)")
    print("=" * 100)
    print(f"{'scenario':<25} {'total':>7} {'Shew':>6} {'EWMA':>6} {'T²':>6} {'SPE':>6} {'Trend':>7} {'Reg':>6}")
    print("-" * 100)
    for _, r in raw_summary.iterrows():
        print(f"{r['scenario']:<25} {r['total_alerts']:>7d} "
              f"{r['shewhart']:>6d} {r['ewma']:>6d} {r['pca_t2']:>6d} "
              f"{r['pca_spe']:>6d} {r['trend']:>7d} {r['regime']:>6d}")

    print("\n" + "=" * 100)
    print("EVENT SESSIONS (operator-facing view)")
    print("=" * 100)
    print(f"{'scenario':<25} {'total':>7} {'Shew':>6} {'EWMA':>6} {'T²':>6} {'SPE':>6} {'Trend':>7} {'Reg':>6}")
    print("-" * 100)
    for _, r in sess_summary.iterrows():
        print(f"{r['scenario']:<25} {r['total_sessions']:>7d} "
              f"{r['shewhart']:>6d} {r['ewma']:>6d} {r['pca_t2']:>6d} "
              f"{r['pca_spe']:>6d} {r['trend']:>7d} {r['regime']:>6d}")

    print("\n" + "=" * 100)
    print("TIME-TO-FIRST-SESSION after known event (minutes; '—' = no event or no detection)")
    print("=" * 100)
    print(f"{'scenario':<25} {'event(h)':>9} {'Shew':>7} {'EWMA':>7} {'T²':>7} {'SPE':>7} {'Trend':>7} {'Reg':>7}")
    print("-" * 100)
    for _, r in sess_summary.iterrows():
        ev = "—" if pd.isna(r["event_time_h"]) else f"{r['event_time_h']:.0f}"
        cells = [r['ttfs_shewhart_min'], r['ttfs_ewma_min'], r['ttfs_pca_t2_min'],
                 r['ttfs_pca_spe_min'], r['ttfs_trend_min'], r['ttfs_regime_min']]
        cell_strs = ['—'.rjust(7) if pd.isna(c) else f'{c:>7.1f}' for c in cells]
        print(f"{r['scenario']:<25} {ev:>9} " + ' '.join(cell_strs))

    print("\n" + "=" * 100)
    print("EVENT SESSION DETAIL — thermal_drift (top 8 by severity)")
    print("=" * 100)
    _, _, tdf = scenario_results["thermal_drift"]
    if not tdf.empty:
        sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        tdf2 = tdf.copy()
        tdf2["_sev"] = tdf2["peak_severity"].map(sev_order)
        tdf2 = tdf2.sort_values(["_sev", "alert_count"], ascending=[False, False]).head(8)
        for _, s in tdf2.iterrows():
            msg_trunc = (s['message'][:75] + '…') if len(s['message']) > 75 else s['message']
            print(f"  [{s['peak_severity'].upper():<8s}] {s['detector']:<18s} "
                  f"{str(s['tag']):<20s} duration={s['duration_min']:>6.0f} min  "
                  f"alerts={s['alert_count']:>4d}")
            print(f"             → {msg_trunc}")

    # Figure
    n_scn = len(SCENARIO_GROUND_TRUTH)
    fig, axes = plt.subplots(n_scn, 1, figsize=(15, 2.2 * n_scn))
    detector_colors = {
        "SPC.Shewhart":     "#e74c3c",
        "SPC.EWMA":         "#e67e22",
        "PCA.T2":           "#9b59b6",
        "PCA.SPE":          "#8e44ad",
        "Trend.Projection": "#27ae60",
        "Regime.CUSUM":     "#2980b9",
    }
    detector_y = {d: i for i, d in enumerate(detector_colors.keys())}

    for ax, (scn, gt) in zip(axes, SCENARIO_GROUND_TRUTH.items()):
        df, _, sess_df = scenario_results[scn]
        t0 = df["timestamp"].min()
        t_h = (df["timestamp"] - t0).dt.total_seconds() / 3600

        ax2 = ax.twinx()
        ax2.plot(t_h, df["cstr.T_R_C"], color="#bdc3c7", lw=0.7)
        ax2.set_ylabel("T_R [°C]", color="#7f8c8d", fontsize=8)
        ax2.tick_params(axis='y', labelsize=7, colors="#7f8c8d")

        if not sess_df.empty:
            for _, s in sess_df.iterrows():
                det = s["detector"]
                if det not in detector_y:
                    continue
                start_h = (s["start_time"] - t0).total_seconds() / 3600
                end_h = (s["end_time"] - t0).total_seconds() / 3600
                duration_h = max(0.08, end_h - start_h)
                ax.barh(
                    y=detector_y[det], width=duration_h, left=start_h,
                    height=0.6, color=detector_colors[det], alpha=0.85,
                    edgecolor="black", linewidth=0.3,
                )

        if gt["event_time_h"] is not None:
            ax.axvline(gt["event_time_h"], color="red", linestyle="--", lw=1.2, alpha=0.6)
            ax.text(gt["event_time_h"], 5.7, " event",
                    color="red", fontsize=7, ha="left", va="bottom")

        ax.set_yticks(list(detector_y.values()))
        ax.set_yticklabels(list(detector_y.keys()), fontsize=7)
        ax.set_xlim(0, t_h.max())
        ax.set_ylim(-0.5, 6.0)
        n_sessions = len(sess_df) if not sess_df.empty else 0
        ax.set_title(f"{scn} — {gt['description']}  ({n_sessions} event sessions)",
                     fontsize=10, loc="left")
        ax.grid(True, alpha=0.2, axis="x")

    axes[-1].set_xlabel("Time [hours]")
    plt.suptitle("Axion AI — Event Session Timeline by Scenario",
                 fontsize=13, y=1.001)
    plt.tight_layout()
    out_path = RESULTS_DIR / "analytics_sessions.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    print(f"Summaries saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
