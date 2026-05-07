"""
Axion AI - OPC-UA Mock Server
=============================

Serves the simulated scenario CSVs as a real OPC-UA server. This lets us
validate the OPC-UA client + ingestion pipeline end-to-end using the actual
OPC-UA protocol (not just mocked calls), but without needing a real plant.

Data model on the server
------------------------
Objects are organized under:
    /Objects
        /Axion
            /CSTR01
                T_R        (Double)
                T_J        (Double)
                C_A        (Double)
                ... plus F_feed_SP, F_cool_SP (writable)
            /COL01
                PurityB    (Double)
                ... plus RR_SP (writable)

Node IDs
--------
All nodes use namespace 2 and a string identifier (e.g. "ns=2;s=CSTR01.T_R").
This matches `PILOT_TAG_MAP` in integration/tag_map.py.

Replay loop
-----------
The server advances through the CSV at a configurable speed (default 60×
real-time) and updates all nodes on every tick. Writable nodes (setpoints)
are read back from the server's internal state so that a client-side write
immediately reflects back on the next tick's published snapshot — this
simulates a real DCS where operator-adjusted setpoints persist.
"""

from __future__ import annotations
import asyncio
import argparse
from pathlib import Path
import pandas as pd

from asyncua import Server, ua
from axion_logging import get_logger

logger = get_logger(__name__)


# Structure of what columns map to what OPC-UA node names
CSV_TO_NODE = {
    # CSTR — measured variables
    "cstr.T_R_C":      ("CSTR01", "T_R"),
    "cstr.T_J_C":      ("CSTR01", "T_J"),
    "cstr.C_A":        ("CSTR01", "C_A"),
    "cstr.conversion": ("CSTR01", "X"),
    "cstr.P_R":        ("CSTR01", "P_R"),
    "cstr.T_feed_C":   ("CSTR01", "T_feed"),
    "cstr.T_cool_in_C":("CSTR01", "T_cool_in"),
    # CSTR — manipulated setpoints (writable)
    "cstr.F_feed":     ("CSTR01", "F_feed_SP"),
    "cstr.F_cool":     ("CSTR01", "F_cool_SP"),

    # Column — measured
    "column.purity_B": ("COL01",  "PurityB"),
    "column.x_D":      ("COL01",  "x_D"),
    "column.x_B_A":    ("COL01",  "x_B_A"),
    "column.T_top_C":  ("COL01",  "T_top"),
    "column.T_bot_C":  ("COL01",  "T_bot"),
    "column.Q_reb_kW": ("COL01",  "Q_reb"),
    "column.F_vap_kgh":("COL01",  "F_vap"),
    "column.P_top_bar":("COL01",  "P_top"),
    "column.P_bot_bar":("COL01",  "P_bot"),
    # Column — manipulated
    "column.RR":       ("COL01",  "RR_SP"),
}

# Setpoint columns: the server holds the "true" setpoint value which a client
# can write to. At each tick we do NOT overwrite these from the CSV — we
# preserve any client write. Everything else is refreshed from the CSV.
WRITABLE_COLS = {"cstr.F_feed", "cstr.F_cool", "column.RR"}


async def build_server(endpoint: str) -> tuple[Server, dict]:
    """
    Create the OPC-UA server, publish the object hierarchy, and return the
    server + a dict of node references keyed by Axion tag.
    """
    server = Server()
    await server.init()
    server.set_endpoint(endpoint)
    server.set_server_name("Axion AI Mock OPC-UA Server")

    # Anonymous access + no security (MVP — a real server would use certs)
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])

    # Custom namespace
    ns_idx = await server.register_namespace("http://axion.ai/mock")
    assert ns_idx == 2, f"Expected ns=2, got {ns_idx}"

    objects = server.nodes.objects
    axion = await objects.add_object(ns_idx, "Axion")

    # Create sub-objects for each unit
    unit_nodes: dict[str, any] = {}
    for unit_name in {v[0] for v in CSV_TO_NODE.values()}:
        unit_nodes[unit_name] = await axion.add_object(ns_idx, unit_name)

    # Create one variable per mapped column
    tag_to_node = {}
    for csv_col, (unit, var_name) in CSV_TO_NODE.items():
        node = await unit_nodes[unit].add_variable(
            f"ns={ns_idx};s={unit}.{var_name}",
            var_name,
            0.0,                       # initial value
            varianttype=ua.VariantType.Double,
        )
        if csv_col in WRITABLE_COLS:
            await node.set_writable()
        tag_to_node[csv_col] = node

    # A dedicated node that exposes the current simulated time as epoch seconds.
    # Clients that want to preserve scenario cadence (vs wall-clock) can read
    # this and use it as the sample timestamp.
    sim_time_node = await axion.add_variable(
        f"ns={ns_idx};s=SIM_TIME",
        "SIM_TIME",
        0.0,
        varianttype=ua.VariantType.Double,
    )
    tag_to_node["__SIM_TIME__"] = sim_time_node

    return server, tag_to_node


async def replay_loop(tag_to_node: dict, df: pd.DataFrame, speed: float):
    """
    Continuously advance through the CSV, writing new values to server nodes.
    Writable (setpoint) nodes preserve the value the client has written.
    Also updates the SIM_TIME node with the current simulated timestamp
    (epoch seconds) so clients can sample against scenario cadence.
    """
    # CSV is sampled at 1 sample per minute. With speed=60x, one wall second
    # = 1 sample.
    samples_per_second = max(1.0, speed / 60.0)
    tick_interval = 1.0 / samples_per_second
    idx = 0
    n = len(df)

    # Cache non-writable node list for efficient refresh
    nonwritable_nodes = [
        (tag, node) for tag, node in tag_to_node.items()
        if tag not in WRITABLE_COLS and tag != "__SIM_TIME__"
    ]
    writable_nodes = [
        (tag, node) for tag, node in tag_to_node.items()
        if tag in WRITABLE_COLS
    ]
    sim_time_node = tag_to_node.get("__SIM_TIME__")

    # First pass: initialize writable setpoints from first CSV row
    first_row = df.iloc[0]
    for tag, node in writable_nodes:
        if tag in df.columns:
            await node.write_value(float(first_row[tag]))

    logger.info("Replay started", extra={
        "samples": n,
        "speed": f"{speed}x",
        "tick_ms": f"{tick_interval*1000:.0f}",
    })

    while True:
        row = df.iloc[idx]
        # Refresh measured variables from the CSV
        for tag, node in nonwritable_nodes:
            if tag in df.columns:
                try:
                    await node.write_value(float(row[tag]))
                except Exception:
                    pass
        # Update simulated time (epoch seconds from the scenario timestamp)
        if sim_time_node is not None:
            try:
                sim_epoch = float(pd.Timestamp(row["timestamp"]).timestamp())
                await sim_time_node.write_value(sim_epoch)
            except Exception:
                pass

        idx = (idx + 1) % n
        await asyncio.sleep(tick_interval)


async def main():
    parser = argparse.ArgumentParser(description="Axion AI mock OPC-UA server")
    parser.add_argument("--endpoint", default="opc.tcp://0.0.0.0:4840",
                        help="OPC-UA server endpoint (default opc.tcp://0.0.0.0:4840)")
    parser.add_argument("--scenario", default="thermal_drift",
                        help="Scenario CSV to replay (default thermal_drift)")
    parser.add_argument("--speed", type=float, default=60.0,
                        help="Replay speed multiplier (default 60x real-time)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing scenario CSVs (defaults to project data/)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parents[1] / "data"
    csv_path = data_dir / f"{args.scenario}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Scenario CSV not found: {csv_path}")

    logger.info("Loading scenario", extra={"scenario": args.scenario})
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    logger.info("Starting OPC-UA server", extra={"endpoint": args.endpoint})
    server, tag_to_node = await build_server(args.endpoint)

    async with server:
        example_nodes = {tag: str(tag_to_node[tag].nodeid)
                         for tag in list(tag_to_node.keys())[:3]}
        logger.info("Server online", extra={
            "tags": len(tag_to_node),
            "example_nodes": example_nodes,
        })
        await replay_loop(tag_to_node, df, args.speed)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
