"""
Axion AI — Generate batch reactor scenario CSVs
================================================

Produces three CSVs under data/ for the BATCH_PROFILE:

    batch_normal.csv         Nominal cooling, well-controlled exotherm.
    batch_runaway.csv        Coolant pump trips at t=60min → temperature spikes.
    batch_slow_kinetics.csv  Reduced k0 → slow conversion, low product yield.

Run:
    python scripts/generate_batch_scenarios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from simulator.batch_reactor import BatchParams, simulate_batch


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def write(df: pd.DataFrame, name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def scenario_normal() -> pd.DataFrame:
    return simulate_batch(BatchParams())


def scenario_runaway() -> pd.DataFrame:
    return simulate_batch(BatchParams(
        F_cool_schedule={0.0: 0.4, 60.0: 0.05},
    ))


def scenario_slow_kinetics() -> pd.DataFrame:
    return simulate_batch(BatchParams(k0=2.0e7))


def main() -> None:
    out = []
    for name, fn in [
        ("batch_normal",         scenario_normal),
        ("batch_runaway",        scenario_runaway),
        ("batch_slow_kinetics",  scenario_slow_kinetics),
    ]:
        df = fn()
        path = write(df, name)
        out.append((name, len(df), path))
        print(f"  {name:25s}  {len(df):4d} samples  →  {path}")
    print(f"\nGenerated {len(out)} batch scenarios under {DATA_DIR}")


if __name__ == "__main__":
    main()
