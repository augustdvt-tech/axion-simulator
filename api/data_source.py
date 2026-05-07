"""
Axion AI — Live data source switcher
=====================================

The dashboard exposes a single `state.run.process_data` DataFrame that the
analytics, replay clock, and UI all read from. Until now, that DataFrame was
always populated from a scenario CSV.

This module adds a second source — the OPC-UA stream — and the buffering /
projection logic that lets either source feed the same canonical DataFrame.

Design
------
- The selected source is `state.data_source` ∈ {"replay", "opcua"}.
- In `replay` mode, behavior is unchanged (load_scenario fills process_data).
- In `opcua` mode, an `OpcuaBuffer` holds a rolling window of incoming
  Samples and exposes them as a DataFrame with the canonical column shape
  the rest of the system already understands.
- The replay loop checks the active source on each tick and either advances
  through the static CSV (replay) or rebuilds `state.run.process_data` from
  the live buffer (opcua).

This module is deliberately small and synchronous. The OPC-UA samples
themselves arrive on the asyncio event loop via `IntegrationService`; we
just append them to a thread-safe deque here.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional

import pandas as pd

from profile import active_profile


def get_live_columns() -> List[str]:
    """Active profile's canonical column shape (timestamp + tags)."""
    return list(active_profile().live_columns)


# Module-level constant for the active profile at import time. Code that
# wants the up-to-date list (after a profile switch) should call
# `get_live_columns()` instead of relying on this.
LIVE_COLUMNS: List[str] = get_live_columns()

# How many samples to keep in the live ring buffer. 1 sample/sec × 4h ≈ 14400.
DEFAULT_BUFFER_CAPACITY = 14_400


@dataclass
class _BufferedSample:
    timestamp: pd.Timestamp
    values: dict   # axion_tag -> float


class OpcuaBuffer:
    """Thread-safe rolling buffer of OPC-UA samples projected onto the
    canonical Axion schema.

    `append(sample)` is called from the asyncio event loop where the
    `IntegrationService` runs. `to_dataframe()` is called from the replay
    loop. Both run in the same event loop so contention is mild, but the
    lock is cheap and makes the contract explicit.
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_CAPACITY):
        self.capacity = capacity
        self._buf: Deque[_BufferedSample] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def append(self, sample) -> None:
        """Append a Sample object (from integration.opcua_source.Sample).

        Unknown tags are dropped; missing canonical tags are filled with NaN
        when the DataFrame is materialized. The timestamp is the Sample's
        Unix timestamp (float seconds), converted to pd.Timestamp here.
        """
        try:
            ts = pd.to_datetime(float(sample.timestamp), unit="s")
        except Exception:
            ts = pd.Timestamp.utcnow()
        # Project Sample.values onto canonical columns (profile-aware)
        live_cols = set(get_live_columns())
        row: dict = {}
        for tag, val in (sample.values or {}).items():
            if tag in live_cols:
                row[tag] = float(val)
        with self._lock:
            self._buf.append(_BufferedSample(timestamp=ts, values=row))

    def append_many(self, samples: Iterable) -> None:
        for s in samples:
            self.append(s)

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize the buffer as a canonical-shape DataFrame.

        Missing columns are NaN-filled. Empty buffer returns an empty
        DataFrame with the right columns so downstream code keeps working.
        """
        with self._lock:
            snapshot = list(self._buf)
        live_cols = get_live_columns()
        if not snapshot:
            return pd.DataFrame(columns=live_cols)
        rows = []
        for s in snapshot:
            row = {"timestamp": s.timestamp}
            for col in live_cols[1:]:
                row[col] = s.values.get(col, float("nan"))
            rows.append(row)
        return pd.DataFrame(rows, columns=live_cols)
