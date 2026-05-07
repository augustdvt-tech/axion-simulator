"""
Axion AI — Industrial integration service
==========================================

Wraps `OPCUASource` with a long-running asyncio task that publishes its
status (connected / disconnected / sample counters / last error) to the
rest of the application.

This is what the FastAPI server uses to bridge live plant data into the
canonical Axion tag stream — independently of the simulator-driven replay.
The simulator and the OPC-UA source are not exclusive: a deployment can
run with `AXION_OPCUA_ENABLED=true` to listen to a real PLC while still
using the simulator data for offline experiments.

Configuration via environment:
    AXION_OPCUA_ENABLED      "true"  to start the source on server startup
    AXION_OPCUA_TAG_MAP      Path to a YAML/JSON tag map file. If unset,
                             falls back to PILOT_TAG_MAP.
    AXION_OPCUA_ENDPOINT     Override the endpoint URL from the tag map
                             (useful for swapping prod ↔ staging endpoints
                             without editing the file)
    AXION_OPCUA_USERNAME     Override the username
    AXION_OPCUA_PASSWORD     Override the password
    AXION_OPCUA_SECURITY     Override security_policy (None | Basic256Sha256)
    AXION_OPCUA_CERT_PATH    Path to client certificate (DER/PEM)
    AXION_OPCUA_KEY_PATH     Path to private key (PEM)
    AXION_OPCUA_TIME_NODE    Node ID exposing simulated time (optional)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from .opcua_source import OPCUASource, Sample
from .tag_map import PILOT_TAG_MAP, TagMap


@dataclass
class IntegrationStatus:
    """Lightweight, JSON-serializable health snapshot for the dashboard."""
    enabled: bool = False
    connected: bool = False
    endpoint: Optional[str] = None
    last_sample_ts: Optional[float] = None
    samples_received: int = 0
    last_error: Optional[str] = None
    n_tags: int = 0
    started_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled":          self.enabled,
            "connected":        self.connected,
            "endpoint":         self.endpoint,
            "last_sample_ts":   self.last_sample_ts,
            "samples_received": self.samples_received,
            "last_error":       self.last_error,
            "n_tags":           self.n_tags,
            "started_at":       self.started_at,
        }


def _override_from_env(tag_map: TagMap) -> TagMap:
    """Apply environment-variable overrides on top of a parsed tag map."""
    srv = tag_map.server
    if (e := os.environ.get("AXION_OPCUA_ENDPOINT", "").strip()):
        srv.endpoint = e
    if (u := os.environ.get("AXION_OPCUA_USERNAME", "").strip()):
        srv.username = u
    if (p := os.environ.get("AXION_OPCUA_PASSWORD", "").strip()):
        srv.password = p
    if (s := os.environ.get("AXION_OPCUA_SECURITY", "").strip()):
        srv.security_policy = s
    if (c := os.environ.get("AXION_OPCUA_CERT_PATH", "").strip()):
        srv.cert_path = c
    if (k := os.environ.get("AXION_OPCUA_KEY_PATH", "").strip()):
        srv.key_path = k
    return tag_map


def load_tag_map_from_env(default: TagMap = PILOT_TAG_MAP) -> TagMap:
    """Load a TagMap based on env vars. Falls back to the pilot map.

    Order of precedence:
      1. AXION_OPCUA_TAG_MAP (path) → parse from file
      2. default argument (in-memory PILOT_TAG_MAP)
      Then env-var overrides are applied to the chosen map.
    """
    path = os.environ.get("AXION_OPCUA_TAG_MAP", "").strip()
    if path:
        tag_map = TagMap.from_file(Path(path))
    else:
        tag_map = default
    return _override_from_env(tag_map)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


class IntegrationService:
    """Long-running OPC-UA bridge. Lifecycle: start() → run forever → stop()."""

    def __init__(
        self,
        tag_map: Optional[TagMap] = None,
        on_sample: Optional[Callable[[Sample], Awaitable[None]]] = None,
        time_node_id: Optional[str] = None,
    ):
        self.tag_map = tag_map or load_tag_map_from_env()
        self.time_node_id = time_node_id or (
            os.environ.get("AXION_OPCUA_TIME_NODE", "").strip() or None
        )
        self._on_sample = on_sample
        self.status = IntegrationStatus(
            endpoint=self.tag_map.server.endpoint,
            n_tags=len(self.tag_map.tags),
        )
        self._source: Optional[OPCUASource] = None
        self._task: Optional[asyncio.Task] = None

    @classmethod
    def from_env(
        cls,
        on_sample: Optional[Callable[[Sample], Awaitable[None]]] = None,
    ) -> Optional["IntegrationService"]:
        """Build a service from env. Returns None if AXION_OPCUA_ENABLED is unset."""
        if not _is_truthy(os.environ.get("AXION_OPCUA_ENABLED", "")):
            return None
        return cls(on_sample=on_sample)

    # ---- callbacks wired into the source ----

    async def _handle_sample(self, sample: Sample) -> None:
        self.status.connected = True
        self.status.samples_received += 1
        self.status.last_sample_ts = sample.timestamp
        self.status.last_error = None
        if self._on_sample is not None:
            try:
                await self._on_sample(sample)
            except Exception as e:
                self.status.last_error = f"on_sample callback: {e}"

    async def _handle_event(self, kind: str, data: dict) -> None:
        if kind == "connected":
            self.status.connected = True
            self.status.last_error = None
        elif kind in ("error", "read_error", "resolve_error"):
            self.status.connected = False
            self.status.last_error = data.get("error") or kind

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self.status.enabled = True
        self.status.started_at = time.time()
        self._source = OPCUASource(
            tag_map=self.tag_map,
            on_sample=self._handle_sample,
            on_event=self._handle_event,
            time_node_id=self.time_node_id,
        )
        self._task = asyncio.create_task(self._source.run())

    async def stop(self) -> None:
        if self._source is not None:
            self._source.stop()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self.status.enabled = False
        self.status.connected = False
