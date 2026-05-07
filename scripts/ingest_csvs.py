#!/usr/bin/env python3
"""
Axion AI — CSV ingestion script.

Reads all scenario CSVs from data/ and bulk-inserts them into the
process_samples hypertable in TimescaleDB.

Usage:
    python scripts/ingest_csvs.py                 # ingest all CSVs, skip existing
    python scripts/ingest_csvs.py --force         # delete + re-ingest all
    python scripts/ingest_csvs.py normal thermal_drift  # specific scenarios only
    python scripts/ingest_csvs.py --data-dir /path/to/data

Requirements:
    AXION_DB_URL env var (or set in .env):
        postgresql://axion:axion@localhost:5432/axion

    Database must be running and migrations must have been applied:
        docker compose up -d
        AXION_DB_URL=... python -m alembic -c db/alembic.ini upgrade head
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Column mapping: CSV tag (dot notation) → DB column (underscore notation)
# ---------------------------------------------------------------------------

TAG_TO_COL: dict[str, str] = {
    "cstr.T_R_C":        "cstr_T_R_C",
    "cstr.T_J_C":        "cstr_T_J_C",
    "cstr.C_A":          "cstr_C_A",
    "cstr.F_feed":       "cstr_F_feed",
    "cstr.F_cool":       "cstr_F_cool",
    "cstr.T_feed_C":     "cstr_T_feed_C",
    "cstr.T_cool_in_C":  "cstr_T_cool_in_C",
    "cstr.P_R":          "cstr_P_R",
    "cstr.conversion":   "cstr_conversion",
    "column.x_D":        "column_x_D",
    "column.x_B_A":      "column_x_B_A",
    "column.purity_B":   "column_purity_B",
    "column.T_top_C":    "column_T_top_C",
    "column.T_bot_C":    "column_T_bot_C",
    "column.RR":         "column_RR",
    "column.F_vap_kgh":  "column_F_vap_kgh",
    "column.Q_reb_kW":   "column_Q_reb_kW",
    "column.P_top_bar":  "column_P_top_bar",
    "column.P_bot_bar":  "column_P_bot_bar",
}

DB_COLS = ["timestamp", "scenario"] + list(TAG_TO_COL.values())
BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection(db_url: str) -> psycopg2.extensions.connection:
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = False
        return conn
    except psycopg2.OperationalError as e:
        print(f"[ingest] ERROR: cannot connect to database.\n  {e}")
        print("[ingest] Is TimescaleDB running?  docker compose up -d")
        sys.exit(1)


def scenario_exists(cur, scenario: str) -> bool:
    cur.execute("SELECT 1 FROM scenarios WHERE name = %s", (scenario,))
    return cur.fetchone() is not None


def delete_scenario(cur, scenario: str) -> int:
    cur.execute("DELETE FROM process_samples WHERE scenario = %s", (scenario,))
    deleted = cur.rowcount
    cur.execute("DELETE FROM scenarios WHERE name = %s", (scenario,))
    return deleted


def load_csv(path: Path, scenario: str) -> list[tuple]:
    df = pd.read_csv(path, parse_dates=["timestamp"])

    rows: list[tuple] = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        values = [ts, scenario]
        for tag in TAG_TO_COL:
            values.append(float(row[tag]) if tag in row and pd.notna(row[tag]) else None)
        rows.append(tuple(values))

    return rows


def ingest_scenario(conn, scenario: str, csv_path: Path, force: bool) -> None:
    t0 = time.time()
    with conn.cursor() as cur:
        if scenario_exists(cur, scenario):
            if not force:
                print(f"  {scenario:<25s} — already ingested, skipping (use --force to re-ingest)")
                return
            deleted = delete_scenario(cur, scenario)
            print(f"  {scenario:<25s} — deleted {deleted} existing rows, re-ingesting…")

        rows = load_csv(csv_path, scenario)
        n = len(rows)

        insert_sql = (
            f"INSERT INTO process_samples ({', '.join(DB_COLS)}) VALUES %s "
            f"ON CONFLICT (timestamp, scenario) DO NOTHING"
        )
        for i in range(0, n, BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            psycopg2.extras.execute_values(cur, insert_sql, batch)

        ingested_at = datetime.now(tz=timezone.utc)
        cur.execute(
            "INSERT INTO scenarios (name, n_samples, ingested_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET n_samples = EXCLUDED.n_samples, "
            "ingested_at = EXCLUDED.ingested_at",
            (scenario, n, ingested_at),
        )

    conn.commit()
    elapsed = time.time() - t0
    print(f"  {scenario:<25s} — {n:>5} samples  {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest scenario CSVs into TimescaleDB")
    parser.add_argument(
        "scenarios", nargs="*",
        help="Scenario names to ingest (default: all CSVs in data-dir)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing rows for the scenario before re-inserting",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing scenario CSVs (default: <project>/data/)",
    )
    parser.add_argument(
        "--db-url", default=None,
        help="PostgreSQL connection URL (default: AXION_DB_URL env var)",
    )
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("AXION_DB_URL")
    if not db_url:
        print("[ingest] ERROR: no database URL. Set AXION_DB_URL or pass --db-url.")
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data"

    if not data_dir.exists():
        print(f"[ingest] ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    # Resolve CSV paths
    if args.scenarios:
        csv_files = []
        for name in args.scenarios:
            p = data_dir / f"{name}.csv"
            if not p.exists():
                print(f"[ingest] WARNING: {p} not found, skipping.")
            else:
                csv_files.append((name, p))
    else:
        csv_files = sorted(
            (p.stem, p) for p in data_dir.glob("*.csv")
        )

    if not csv_files:
        print("[ingest] No CSV files found. Nothing to do.")
        sys.exit(0)

    print(f"[ingest] Connecting to {db_url.split('@')[-1]} …")
    conn = get_connection(db_url)

    print(f"[ingest] Ingesting {len(csv_files)} scenario(s) "
          f"{'(force mode)' if args.force else '(skip existing)'}:")

    total_t0 = time.time()
    for scenario, path in csv_files:
        try:
            ingest_scenario(conn, scenario, path, force=args.force)
        except Exception as e:
            conn.rollback()
            print(f"  {scenario:<25s} — ERROR: {e}")

    conn.close()
    total_elapsed = time.time() - total_t0
    print(f"\n[ingest] Done in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    main()
