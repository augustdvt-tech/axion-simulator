"""
Axion AI - Ingestion Service
============================

Bridges the OPC-UA source to the downstream Axion pipeline. Accumulates
incoming samples into a rolling DataFrame buffer, periodically invokes the
analytical engine on the buffer, and pushes any new event sessions /
recommendations through a registered callback.

The service runs in two buffer modes:

1. ROLLING WINDOW (default) — keeps the last N hours of samples in memory.
   On each evaluation tick, runs the analytics pipeline over the buffer to
   generate fresh sessions and recommendations.

2. INCREMENTAL (future) — incremental SPC/PCA updates without re-running
   over the whole window. Not implemented in the MVP because the current
   analytics are fast enough (~100ms for 24h of data) that rolling window
   is simpler and sufficient.

Design principle
----------------
The analytical engine was built for batch evaluation over CSVs. Rather than
re-architect it for streaming, we run it against a rolling pandas DataFrame
that grows as samples arrive. This keeps the engine code unchanged across
the simulator-driven and live-data deployments.

A side benefit: analytics results are identical between batch replay and
live streaming, which makes regression testing trivial.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Awaitable, List, Optional
import pandas as pd

from .opcua_source import Sample
from analytics import AnalyticalEngine
from recommendations import RecommendationEngine
from consensus import ConsensusController
from axion_logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionConfig:
    window_hours: float = 4.0            # rolling window to evaluate
    evaluation_interval_s: float = 10.0   # how often to run analytics
    min_samples_before_eval: int = 30     # warm-up threshold
    column_order: Optional[List[str]] = None   # enforce canonical column order


class IngestionService:
    """
    Consumes Sample objects from an OPC-UA source and feeds them into the
    Axion analytical + recommendation + consensus pipeline.

    Usage:
        service = IngestionService(ae_engine, re_engine, cc_controller)
        async def on_sample(sample: Sample):
            await service.handle_sample(sample)
        # Pass on_sample to OPCUASource as its callback.

        # Then let the service run its evaluation loop in the background:
        asyncio.create_task(service.run())
    """

    def __init__(
        self,
        ae: AnalyticalEngine,
        re: RecommendationEngine,
        cc: Optional[ConsensusController] = None,
        config: Optional[IngestionConfig] = None,
        on_new_recommendations: Optional[Callable[[list], Awaitable[None]]] = None,
        on_new_sample: Optional[Callable[[Sample], Awaitable[None]]] = None,
    ):
        self.ae = ae
        self.re = re
        self.cc = cc
        self.config = config or IngestionConfig()
        self.on_new_recommendations = on_new_recommendations
        self.on_new_sample = on_new_sample

        # Rolling buffer of samples. For MVP simplicity we keep it as a list
        # and convert to DataFrame on each evaluation. Memory usage is fine
        # at 1Hz for 4 hours = 14k rows.
        self._buffer: list[dict] = []
        # Known recommendation IDs so we only emit new ones
        self._seen_rec_ids: set[str] = set()
        self._stop = False

    # ---- sample handling ----

    async def handle_sample(self, sample: Sample) -> None:
        """Called by the OPC-UA source for every incoming sample."""
        ts = datetime.fromtimestamp(sample.timestamp, tz=timezone.utc).replace(tzinfo=None)
        row = {"timestamp": ts}
        row.update(sample.values)
        self._buffer.append(row)

        # Trim the buffer to the rolling window
        if self._buffer:
            cutoff = ts - pd.Timedelta(hours=self.config.window_hours)
            # Keep buffer a pure list for speed; binary trim from the front
            while self._buffer and self._buffer[0]["timestamp"] < cutoff:
                self._buffer.pop(0)

        if self.on_new_sample is not None:
            await self.on_new_sample(sample)

    # ---- evaluation loop ----

    async def run(self) -> None:
        while not self._stop:
            try:
                await self.evaluate_once()
            except Exception as e:
                # Don't let evaluation errors kill the loop
                logger.error("Evaluation error", extra={"error": str(e)})
            await asyncio.sleep(self.config.evaluation_interval_s)

    def stop(self) -> None:
        self._stop = True

    async def evaluate_once(self) -> None:
        if len(self._buffer) < self.config.min_samples_before_eval:
            return

        # Build DataFrame from the buffer
        df = pd.DataFrame(self._buffer)
        # Enforce presence of required columns (fill missing with NaN)
        if self.config.column_order:
            for col in self.config.column_order:
                if col not in df.columns:
                    df[col] = float("nan")

        # Run analytics → recommendations → (optional) consensus
        sessions = self.ae.run_sessions(df)
        recs = self.re.generate(sessions, df)

        # Filter to recommendations we haven't seen yet
        new_recs = [r for r in recs if r.id not in self._seen_rec_ids]
        for r in new_recs:
            self._seen_rec_ids.add(r.id)

        if new_recs and self.on_new_recommendations is not None:
            await self.on_new_recommendations(new_recs)

        # If a consensus controller is attached, process any new recommendations
        # through the consensus loop.
        if self.cc is not None and new_recs:
            self.cc.process(new_recs, df)

    # ---- introspection ----

    def buffer_summary(self) -> dict:
        if not self._buffer:
            return {"samples": 0, "tags": 0, "oldest": None, "newest": None}
        return {
            "samples":  len(self._buffer),
            "tags":     len(self._buffer[-1]) - 1,   # minus timestamp
            "oldest":   self._buffer[0]["timestamp"].isoformat(),
            "newest":   self._buffer[-1]["timestamp"].isoformat(),
        }
