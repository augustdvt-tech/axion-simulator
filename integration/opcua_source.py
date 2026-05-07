"""
Axion AI - OPC-UA Source Client
===============================

Reads tag values from any standard OPC-UA server on a configurable polling
interval. Produces a stream of samples in the canonical Axion format, ready
to be fed into the AnalyticalEngine.

Design:
- Polls all configured tags together on each interval (single subscription
  could be added later for efficiency; for MVP polling is simpler and more
  robust to reconnection).
- Validates each value against its configured min/max range; out-of-range
  values are marked with quality_flag != 0 but still forwarded so the
  analytics can see them.
- Reconnects automatically if the OPC-UA session drops.
- Emits samples via an async callback so the consumer (IngestionService)
  can write to TimescaleDB, push WebSocket events, or feed the engine
  incrementally.

Quality flag semantics (compatible with the simulator's DataLogger)
    0 = GOOD
    1 = OUT_OF_RANGE
    2 = STALE (value hasn't changed for >N samples)
    3 = READ_ERROR
    4 = NOT_CONNECTED

Time stamps
-----------
By default, samples are timestamped with `time.time()` (wall-clock UTC).
For simulator-backed deployments where the process runs faster than real
time, the source can also fetch the simulated timestamp from a designated
node ID — this is what `time_node_id` enables. When set, every sample is
stamped with the value read from that node (in epoch seconds).
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Awaitable, List
import time

from asyncua import Client, ua
from .tag_map import TagMap, TagMapping


@dataclass
class Sample:
    """A single sample emitted by the OPC-UA source."""
    timestamp: float                         # Unix timestamp (seconds)
    values: Dict[str, float] = field(default_factory=dict)   # axion_tag -> value
    quality: Dict[str, int] = field(default_factory=dict)    # axion_tag -> quality flag
    source: str = "opcua"


class OPCUASource:
    """
    OPC-UA client that polls configured tags and emits samples.

    Usage:
        source = OPCUASource(tag_map, on_sample=my_callback)
        await source.run()          # runs until stopped

    The callback signature is:
        async def on_sample(sample: Sample) -> None
    """

    def __init__(
        self,
        tag_map: TagMap,
        on_sample: Callable[[Sample], Awaitable[None]],
        on_event: Optional[Callable[[str, dict], Awaitable[None]]] = None,
        time_node_id: Optional[str] = None,
    ):
        self.tag_map = tag_map
        self.on_sample = on_sample
        self.on_event = on_event or self._noop_event
        self.time_node_id = time_node_id

        self._client: Optional[Client] = None
        self._node_refs: Dict[str, any] = {}   # axion_tag -> asyncua Node
        self._time_node_ref = None
        self._stop = False
        # Staleness tracking
        self._last_value: Dict[str, float] = {}
        self._last_change_ts: Dict[str, float] = {}

    async def _noop_event(self, kind: str, data: dict) -> None:
        pass

    # ---- lifecycle ----

    async def run(self) -> None:
        """Connect → poll forever → reconnect on failure → until stop() called."""
        while not self._stop:
            try:
                await self._connect()
                await self.on_event("connected", {"endpoint": self.tag_map.server.endpoint})
                await self._poll_loop()
            except Exception as e:
                await self.on_event("error", {"error": str(e)})
                await asyncio.sleep(2.0)
            finally:
                await self._disconnect()
                if not self._stop:
                    # Reconnect after a short delay
                    await asyncio.sleep(2.0)

    def stop(self) -> None:
        self._stop = True

    # ---- connection ----

    async def _connect(self) -> None:
        cfg = self.tag_map.server
        self._client = Client(url=cfg.endpoint, timeout=cfg.connect_timeout_s)
        if cfg.username:
            self._client.set_user(cfg.username)
            if cfg.password:
                self._client.set_password(cfg.password)
        # Apply security policy beyond None when cert paths are provided.
        # Format expected by asyncua:
        #   "Basic256Sha256,SignAndEncrypt,/path/to/cert.der,/path/to/key.pem"
        sec = (cfg.security_policy or "None").strip()
        if sec and sec.lower() != "none":
            cert_path = getattr(cfg, "cert_path", None)
            key_path  = getattr(cfg, "key_path",  None)
            mode      = getattr(cfg, "security_mode", "SignAndEncrypt")
            if cert_path and key_path:
                await self._client.set_security_string(
                    f"{sec},{mode},{cert_path},{key_path}"
                )
        await self._client.connect()
        await self._resolve_nodes()

    async def _disconnect(self) -> None:
        try:
            if self._client is not None:
                await self._client.disconnect()
        except Exception:
            pass
        self._client = None
        self._node_refs.clear()

    async def _resolve_nodes(self) -> None:
        """Pre-resolve node references for all configured tags."""
        assert self._client is not None
        for mapping in self.tag_map.tags:
            try:
                node = self._client.get_node(mapping.node_id)
                self._node_refs[mapping.axion_tag] = node
            except Exception as e:
                await self.on_event("resolve_error", {
                    "tag": mapping.axion_tag, "node_id": mapping.node_id,
                    "error": str(e),
                })
        # Resolve optional simulated-time node
        if self.time_node_id:
            try:
                self._time_node_ref = self._client.get_node(self.time_node_id)
            except Exception as e:
                await self.on_event("resolve_error", {
                    "tag": "__SIM_TIME__", "node_id": self.time_node_id,
                    "error": str(e),
                })

    # ---- polling ----

    async def _poll_loop(self) -> None:
        interval_s = self.tag_map.sampling.interval_ms / 1000.0
        while not self._stop:
            try:
                sample = await self._read_all()
                await self.on_sample(sample)
            except Exception as e:
                await self.on_event("read_error", {"error": str(e)})
                # Force reconnect by breaking out of the loop
                raise
            await asyncio.sleep(interval_s)

    async def _read_all(self) -> Sample:
        now = time.time()
        # If a simulated-time node is configured, read it and use instead
        if self._time_node_ref is not None:
            try:
                sim_ts = await self._time_node_ref.read_value()
                if sim_ts is not None:
                    now = float(sim_ts)
            except Exception:
                pass   # fall back to wall-clock
        sample = Sample(timestamp=now)
        stale_threshold = 10.0   # seconds without change → mark stale

        # Read nodes in parallel for efficiency
        tasks = []
        tags_ordered: List[TagMapping] = []
        for mapping in self.tag_map.tags:
            node = self._node_refs.get(mapping.axion_tag)
            if node is None:
                continue
            tags_ordered.append(mapping)
            tasks.append(self._safe_read(node))

        results = await asyncio.gather(*tasks, return_exceptions=False)

        for mapping, value in zip(tags_ordered, results):
            if value is None:
                sample.quality[mapping.axion_tag] = 3   # READ_ERROR
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                sample.quality[mapping.axion_tag] = 3
                continue

            # Range check
            quality = 0
            if mapping.min_range is not None and v < mapping.min_range:
                quality = 1
            elif mapping.max_range is not None and v > mapping.max_range:
                quality = 1

            # Staleness check (value hasn't moved)
            prev = self._last_value.get(mapping.axion_tag)
            if prev is not None and abs(prev - v) < 1e-9:
                last_change = self._last_change_ts.get(mapping.axion_tag, now)
                if (now - last_change) > stale_threshold and quality == 0:
                    quality = 2
            else:
                self._last_change_ts[mapping.axion_tag] = now
            self._last_value[mapping.axion_tag] = v

            sample.values[mapping.axion_tag] = v
            sample.quality[mapping.axion_tag] = quality

        return sample

    async def _safe_read(self, node) -> Optional[float]:
        try:
            return await node.read_value()
        except Exception:
            return None
