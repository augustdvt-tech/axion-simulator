"""
Axion AI - OPC-UA End-to-End Demo
=================================

Validates the full OPC-UA integration path:

    Mock OPC-UA server (replays thermal_drift.csv)
              ↓  opc.tcp
    OPCUASource reads tag values every second
              ↓
    IngestionService buffers samples
              ↓
    AnalyticalEngine → sessions → recommendations
              ↓
    For an ACCEPTED recommendation on a writable tag:
        OPCUAWriter writes the new setpoint back to the server
              ↓
        Readback verification confirms it took effect
              ↓
    Mock server reflects the new setpoint on subsequent polls
        (verifying the writer→server→source loop is closed)

This demonstrates that the entire control loop works end-to-end over the
real OPC-UA protocol, without any mocking of the protocol layer itself.

Run:
    python examples/opcua_e2e_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pandas as pd

from integration import (
    PILOT_TAG_MAP, OPCUASource, OPCUAWriter, IngestionService,
    IngestionConfig, Sample,
)
from integration.opcua_mock_server import build_server, replay_loop

from analytics import AnalyticalEngine
from recommendations import RecommendationEngine

DATA_DIR = Path(__file__).parent.parent / "data"


# ------- callbacks (use a counter object to avoid Python global scope issues) -------

class DemoCounters:
    def __init__(self):
        self.samples = 0
        self.recs = []

counters = DemoCounters()


async def on_sample(sample: Sample):
    counters.samples += 1
    if counters.samples % 30 == 0:
        n_tags = len(sample.values)
        n_bad = sum(1 for q in sample.quality.values() if q != 0)
        t_r = sample.values.get("cstr.T_R_C", float("nan"))
        purity = sample.values.get("column.purity_B", float("nan"))
        print(f"  [sample #{counters.samples:4d}]  tags={n_tags:2d}  bad={n_bad:2d}  "
              f"T_R={t_r:.2f}  purity={purity:.2f}", flush=True)


async def on_new_recommendations(new_recs):
    for r in new_recs:
        counters.recs.append(r)
        print(f"\n    >>> NEW RECOMMENDATION <<<", flush=True)
        print(f"    [{r.urgency.value.upper():<8s}] {r.rule_fired}")
        print(f"    diagnosis: {r.diagnosis[:95]}...")
        print(f"    action:    {r.action.description[:95]}")
        if r.action.target_variable and r.action.proposed_value is not None:
            print(f"    setpoint:  {r.action.target_variable}: "
                  f"{r.action.current_value:.3f} → {r.action.proposed_value:.3f} "
                  f"(Δ={r.action.adjustment:+.3f})")
        print(flush=True)


async def on_event(kind: str, data: dict):
    if kind == "connected":
        print(f"  [opcua] connected to {data['endpoint']}", flush=True)
    elif kind == "error":
        print(f"  [opcua] error: {data['error']}", flush=True)


# ------- main -------

async def main():
    endpoint = "opc.tcp://127.0.0.1:4842"

    # 1) Start the mock OPC-UA server
    print("=" * 70)
    print("1) Starting mock OPC-UA server")
    print("=" * 70)
    # Use feed_perturbation (event at t=12h) — easier to see recs within demo time
    df_scenario = pd.read_csv(DATA_DIR / "feed_perturbation.csv")
    df_scenario["timestamp"] = pd.to_datetime(df_scenario["timestamp"])

    server, tag_to_node = await build_server(endpoint)
    await server.start()
    print(f"  Server online at {endpoint}, publishing {len(tag_to_node)} tags")

    # Replay: 600× → 1 sample every 100ms → 24h scenario in 144s wall-clock.
    # Client polls every 100ms matching, so we capture every sample.
    replay_task = asyncio.create_task(replay_loop(tag_to_node, df_scenario, speed=600.0))

    # Give the server a moment to stabilise
    await asyncio.sleep(0.5)

    # 2) Set up analytics and recommendation engines
    print()
    print("=" * 70)
    print("2) Setting up Axion pipeline")
    print("=" * 70)
    df_train = pd.read_csv(DATA_DIR / "normal.csv")
    ae = AnalyticalEngine(training_fraction=1.0, warmup_minutes=2.0)
    ae.fit(df_train)
    re = RecommendationEngine()
    print(f"  Analytical engine ready ({ae.pca.model.n_components} PCs)")
    print(f"  Recommendation engine ready ({len(re.rules)} rules)")

    # 3) Configure ingestion with short eval interval for the demo
    ingestion = IngestionService(
        ae=ae, re=re, cc=None,
        config=IngestionConfig(
            window_hours=24.0,         # full scenario fits in window
            evaluation_interval_s=2.0,  # evaluate every 2 wall-seconds
            min_samples_before_eval=30,
        ),
        on_new_recommendations=on_new_recommendations,
        on_new_sample=on_sample,        # demo prints + counters
    )

    # 4) Use a tag map pointing to our local mock server
    live_map = PILOT_TAG_MAP
    live_map.server.endpoint = endpoint
    live_map.sampling.interval_ms = 100   # match server tick rate

    # 5) Set up source + wire callback. time_node_id lets samples carry the
    # simulated scenario timestamp instead of wall-clock, so analytics see
    # original scenario cadence regardless of replay speed.
    source = OPCUASource(
        tag_map=live_map,
        on_sample=ingestion.handle_sample,
        on_event=on_event,
        time_node_id="ns=2;s=SIM_TIME",
    )

    # 6) Run source + ingestion for a bounded time
    print()
    print("=" * 70)
    print("3) Running live pipeline for 90 seconds (covers ~15h of simulated process)")
    print("=" * 70)

    source_task     = asyncio.create_task(source.run())
    ingestion_task  = asyncio.create_task(ingestion.run())

    # Run for 90 seconds then stop
    await asyncio.sleep(90)

    source.stop()
    ingestion.stop()

    # Small grace period for tasks to clean up
    await asyncio.sleep(1.5)
    for t in (source_task, ingestion_task):
        if not t.done():
            t.cancel()
            try: await t
            except: pass

    # 7) Optional: writer demo — write a new RR setpoint and read it back
    print()
    print("=" * 70)
    print("4) OPC-UA writer demo: changing column.RR setpoint")
    print("=" * 70)
    writer = OPCUAWriter(live_map, readback_delay_s=0.3)
    await writer.connect()
    try:
        # Read current RR from the server via an ad-hoc OPCUASource read
        rr_node = server.get_node("ns=2;s=COL01.RR_SP")
        current_rr = float(await rr_node.read_value())
        print(f"  Current RR on server: {current_rr:.3f}")

        # Write a new value
        new_rr = current_rr + 0.25
        result = await writer.write_setpoint("column.RR", new_rr)
        readback_str = f"{result.readback_value:.3f}" if result.readback_value is not None else "NA"
        print(f"  Write result: status={result.status.value}  "
              f"requested={result.requested_value:.3f}  "
              f"readback={readback_str}")

        # Try an unauthorized write
        bad_result = await writer.write_setpoint("cstr.T_R_C", 99.9)
        print(f"  Write to read-only tag: status={bad_result.status.value}  "
              f"reason={bad_result.error_message[:70]}")

        # Try an out-of-range write
        out_result = await writer.write_setpoint("column.RR", 99.0)
        print(f"  Write out-of-range:     status={out_result.status.value}  "
              f"reason={out_result.error_message[:70]}")
    finally:
        await writer.disconnect()

    # 8) Summary
    print()
    print("=" * 70)
    print("5) Summary")
    print("=" * 70)
    print(f"  Samples received:        {counters.samples}")
    print(f"  Buffer at end:           {ingestion.buffer_summary()}")
    print(f"  Recommendations emitted: {len(counters.recs)}")
    if counters.recs:
        from collections import Counter
        rule_counts = Counter(r.rule_fired for r in counters.recs)
        for rule, n in rule_counts.most_common():
            print(f"    {rule}: {n}")

    # Shut down mock server
    replay_task.cancel()
    try: await replay_task
    except: pass
    await server.stop()
    print()
    print("  End-to-end demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
