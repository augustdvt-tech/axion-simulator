"""
Axion AI — CSV upload validator
================================

Pure functions to validate a user-provided scenario CSV before it's accepted
as a new scenario. No I/O, no FastAPI imports — keeps the validator unit
testable and reusable from CLI tooling.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# Column contract — must match what the analytical engine and recommendation
# rules expect. Mirror of the columns produced by simulator.runner.
REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "cstr.T_R_C",      "cstr.T_J_C",   "cstr.C_A",
    "cstr.F_feed",     "cstr.F_cool",  "cstr.T_feed_C",
    "cstr.T_cool_in_C", "cstr.P_R",    "cstr.conversion",
    "column.x_D",      "column.x_B_A", "column.purity_B",
    "column.T_top_C",  "column.T_bot_C", "column.RR",
    "column.F_vap_kgh", "column.Q_reb_kW",
    "column.P_top_bar", "column.P_bot_bar",
]

MIN_ROWS = 60          # at least 60 samples (1 hour at 1 sample/min)
MAX_ROWS = 100_000     # guard against pathological uploads
MAX_FILE_BYTES = 25 * 1024 * 1024   # 25 MB hard cap

_NAME_RE = re.compile(r"^[a-z0-9_]{2,40}$")


@dataclass
class ValidationResult:
    ok: bool
    df: Optional[pd.DataFrame] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0


def validate_scenario_name(name: str) -> Optional[str]:
    """Return None if valid, else an error message."""
    if not name:
        return "Scenario name required."
    if not _NAME_RE.match(name):
        return ("Scenario name must be 2–40 chars, lowercase letters, "
                "digits or underscore.")
    return None


def validate_csv_bytes(data: bytes) -> ValidationResult:
    """Parse + validate raw CSV bytes. Never raises."""
    errors: List[str] = []
    warnings: List[str] = []

    if len(data) > MAX_FILE_BYTES:
        return ValidationResult(
            ok=False,
            errors=[f"File too large ({len(data)} bytes > {MAX_FILE_BYTES})."],
        )
    if not data:
        return ValidationResult(ok=False, errors=["Empty file."])

    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        return ValidationResult(ok=False, errors=[f"CSV parse error: {e}"])

    n_rows, n_cols = len(df), len(df.columns)

    # Column presence
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Row count
    if n_rows < MIN_ROWS:
        errors.append(f"Too few rows ({n_rows} < {MIN_ROWS}).")
    if n_rows > MAX_ROWS:
        errors.append(f"Too many rows ({n_rows} > {MAX_ROWS}).")

    # Timestamp parseability
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            errors.append(f"Could not parse 'timestamp' column: {e}")

    # Numeric sanity for present required columns
    for col in REQUIRED_COLUMNS:
        if col == "timestamp" or col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_nan = int(coerced.isna().sum())
        if n_nan == n_rows:
            errors.append(f"Column '{col}' is fully non-numeric.")
        elif n_nan > n_rows * 0.10:
            warnings.append(
                f"Column '{col}' has {n_nan} non-numeric values "
                f"({n_nan / max(1, n_rows):.0%})."
            )
        else:
            df[col] = coerced

    if errors:
        return ValidationResult(
            ok=False, errors=errors, warnings=warnings,
            n_rows=n_rows, n_cols=n_cols,
        )

    return ValidationResult(
        ok=True, df=df, errors=[], warnings=warnings,
        n_rows=n_rows, n_cols=n_cols,
    )
