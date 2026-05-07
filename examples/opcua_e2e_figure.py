"""
Axion AI - OPC-UA End-to-End Benchmark Figure
=============================================

Runs the full OPC-UA loop headless and produces a figure showing:
  - How samples accumulate in the buffer over time
  - When each recommendation is emitted, colored by rule
  - Process variables (T_R, purity) overlaid as backdrop

Output: results/opcua_e2e_figure.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from integration import (
    PILOT_TAG_MAP, OPCUASource, IngestionService, IngestionConfig, Sample,
)
from integration.opcua_mock_server import build_server, replay_loop
from analytics import AnalyticalEngine
from recommendations import RecommendationEngine

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


RULE_COLOR = {
    "R01_ThermalDrift":         "#e67e22",
    "R02_FeedComposition":      "#c0392b",
    "R03_ControllerOscillation":"#8e44ad",
    "R04_ColumnEfficiencyLoss": "#d35400",
    "R05_ExcessReflux":         "#27ae60",
    "R06_PurityDeviation":      "#2980b9",
    "R07_SensorFault":          "#7f8c8d",
    "R08_ProductTransition":    "#16a085",
}


async def run_demo_and_capture():
    """Run the OPC-UA demo and capture sample / recommendation timelines."""
    endpoint = "opc.tcp://127.0.0.1:4843"

    # Capture data for the figure
    sample_history: list[dict] = []      # each: {t, T_R, purity}
    rec_timeline:   list[dict] = []      # each: {t, rule, urgency}

    df_scenario = pd.read_csv(DATA_DIR / "feed_perturbation.csv")
    df_scenario["timestamp"] = pd.to_datetime(df_scenario["timestamp"])

    server, tag_to_node = await build_server(endpoint)
    await server.start()
    replay_task = asyncio.create_task(replay_loop(tag_to_node, df_scenario, speed=600.0))
    await asyncio.sleep(0.5)

    df_train = pd.read_csv(DATA_DIR / "normal.csv")
    ae = AnalyticalEngine(training_fraction=1.0, warmup_minutes=2.0)
    ae.fit(df_train)
    re = RecommendationEngine()

    async def capture_sample(s: Sample):
        sample_history.append({
            "t":      s.timestamp,
            "T_R":    s.values.get("cstr.T_R_C"),
            "purity": s.values.get("column.purity_B"),
        })

    async def capture_recs(new_recs):
        for r in new_recs:
            rec_timeline.append({
                "t":       r.timestamp.timestamp(),
                "rule":    r.rule_fired,
                "urgency": r.urgency.value,
            })

    ingestion = IngestionService(
        ae=ae, re=re, cc=None,
        config=IngestionConfig(
            window_hours=24.0,
            evaluation_interval_s=2.0,
            min_samples_before_eval=30,
        ),
        on_new_recommendations=capture_recs,
        on_new_sample=capture_sample,
    )

    live_map = PILOT_TAG_MAP
    live_map.server.endpoint = endpoint
    live_map.sampling.interval_ms = 100

    source = OPCUASource(
        tag_map=live_map,
        on_sample=ingestion.handle_sample,
        on_event=None,
        time_node_id="ns=2;s=SIM_TIME",
    )

    source_task = asyncio.create_task(source.run())
    ingest_task = asyncio.create_task(ingestion.run())

    print(f"  Running OPC-UA demo for 90 s...")
    await asyncio.sleep(90)

    source.stop()
    ingestion.stop()
    await asyncio.sleep(1.5)
    for t in (source_task, ingest_task, replay_task):
        if not t.done():
            t.cancel()
            try: await t
            except: pass
    await server.stop()

    return sample_history, rec_timeline


def make_figure(sample_history, rec_timeline, out_path):
    if not sample_history:
        print("No samples captured; aborting figure.")
        return

    # Convert timestamps to hours-into-scenario
    t0 = sample_history[0]["t"]
    sh = pd.DataFrame(sample_history)
    sh["t_h"] = (sh["t"] - t0) / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # ---- Top: process variables ----
    ax1 = axes[0]
    ax1.plot(sh["t_h"], sh["T_R"], color="#2980b9", lw=1.0, label="T_R [°C]")
    ax1.set_ylabel("T_R [°C]", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax1.grid(True, alpha=0.25)

    ax1b = ax1.twinx()
    ax1b.plot(sh["t_h"], sh["purity"], color="#e67e22", lw=1.0, label="Purity B [%]")
    ax1b.set_ylabel("Purity B [%]", color="#e67e22")
    ax1b.tick_params(axis="y", labelcolor="#e67e22")
    ax1b.axhline(98.5, color="#c0392b", ls=":", lw=0.8, alpha=0.6)

    ax1.set_title("Axion AI — OPC-UA End-to-End: Live Capture from Mock Server",
                  fontsize=12, loc="left")

    # ---- Bottom: recommendation timeline ----
    ax2 = axes[1]
    if rec_timeline:
        rt = pd.DataFrame(rec_timeline)
        rt["t_h"] = (rt["t"] - t0) / 3600.0
        rules = sorted(rt["rule"].unique())
        rule_y = {r: i for i, r in enumerate(rules)}
        for rule in rules:
            sub = rt[rt["rule"] == rule]
            ax2.scatter(sub["t_h"], [rule_y[rule]] * len(sub),
                        color=RULE_COLOR.get(rule, "#555"), alpha=0.6, s=18)
        ax2.set_yticks(list(rule_y.values()))
        ax2.set_yticklabels(list(rule_y.keys()), fontsize=9)
        ax2.set_ylim(-0.5, len(rules) - 0.5)
    ax2.grid(True, alpha=0.25, axis="x")
    ax2.set_xlabel("Scenario time [hours]")
    ax2.set_title(f"{len(rec_timeline)} recommendations emitted live via ingestion pipeline",
                  fontsize=10, loc="left")

    # Footer annotation: how many samples captured
    fig.text(0.995, 0.005,
             f"{len(sample_history)} samples captured · {len(rec_timeline)} recs · "
             f"scenario span {sh['t_h'].max():.1f} h · live over opc.tcp://",
             ha="right", fontsize=8, color="#555", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"  Figure saved: {out_path}")


async def main():
    sample_history, rec_timeline = await run_demo_and_capture()
    print(f"  Captured {len(sample_history)} samples, {len(rec_timeline)} recs")
    out_path = RESULTS_DIR / "opcua_e2e_figure.png"
    make_figure(sample_history, rec_timeline, out_path)


if __name__ == "__main__":
    asyncio.run(main())
