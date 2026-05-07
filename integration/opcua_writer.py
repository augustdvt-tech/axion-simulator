"""
Axion AI - OPC-UA Writer
========================

Writes setpoints to an OPC-UA server. Used by the ConsensusController when
a recommendation is executed in SEMI_AUTONOMOUS or AUTONOMOUS_SUPERVISED
mode.

Safety guarantees
-----------------
1. Writability whitelist — only tags marked `writable: True` in the tag map
   can be written. A rule that recommends writing to a read-only tag is
   rejected *here*, not silently ignored.

2. Range enforcement — the proposed value must fall within the tag's
   [min_range, max_range] from the tag map. These bounds come from the
   plant's engineering specification, not from Axion's rule authors.

3. Read-back verification — after writing, the client reads the value back
   and confirms it took effect within a configurable tolerance. If the
   read-back value does not match, the write is marked FAILED.

4. Error tolerance — all writes run through an exception boundary; a failed
   write never crashes the system. Callers receive a WriteResult and can
   decide whether to retry, roll back, or escalate.

The ConsensusController treats a FAILED write the same way it treats a
rejected decision: the Execution record records the failure with its error
message, and the outcome tracker will not measure an outcome.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from asyncua import Client, ua
from .tag_map import TagMap, TagMapping


class WriteStatus(str, Enum):
    SUCCESS = "success"
    BLOCKED_NOT_WRITABLE = "blocked_not_writable"
    BLOCKED_OUT_OF_RANGE = "blocked_out_of_range"
    TAG_NOT_FOUND = "tag_not_found"
    NOT_CONNECTED = "not_connected"
    WRITE_ERROR = "write_error"
    READBACK_MISMATCH = "readback_mismatch"


@dataclass
class WriteResult:
    status: WriteStatus
    axion_tag: str
    requested_value: float
    readback_value: Optional[float] = None
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == WriteStatus.SUCCESS


class OPCUAWriter:
    """
    OPC-UA client for writing setpoint values to the DCS.

    Typically shares a Client connection with OPCUASource in production; for
    the MVP it opens its own connection per write session for simplicity.
    """

    def __init__(
        self,
        tag_map: TagMap,
        readback_tolerance: float = 0.02,    # 2% relative tolerance
        readback_delay_s: float = 0.5,       # wait before reading back
    ):
        self.tag_map = tag_map
        self.readback_tolerance = readback_tolerance
        self.readback_delay_s = readback_delay_s
        self._tag_index = {t.axion_tag: t for t in tag_map.tags}
        self._client: Optional[Client] = None

    # ---- lifecycle ----

    async def connect(self) -> None:
        cfg = self.tag_map.server
        self._client = Client(url=cfg.endpoint, timeout=cfg.connect_timeout_s)
        if cfg.username:
            self._client.set_user(cfg.username)
            if cfg.password:
                self._client.set_password(cfg.password)
        await self._client.connect()

    async def disconnect(self) -> None:
        try:
            if self._client is not None:
                await self._client.disconnect()
        finally:
            self._client = None

    # ---- main operation ----

    async def write_setpoint(self, axion_tag: str, value: float) -> WriteResult:
        """
        Write `value` to the OPC-UA node mapped to `axion_tag`, after all
        safety checks. Verifies with a read-back.
        """
        mapping = self._tag_index.get(axion_tag)
        if mapping is None:
            return WriteResult(
                status=WriteStatus.TAG_NOT_FOUND,
                axion_tag=axion_tag,
                requested_value=value,
                error_message=f"Unknown Axion tag: {axion_tag}",
            )
        if not mapping.writable:
            return WriteResult(
                status=WriteStatus.BLOCKED_NOT_WRITABLE,
                axion_tag=axion_tag,
                requested_value=value,
                error_message=(
                    f"Tag {axion_tag} is not marked writable in the tag map. "
                    f"Plant engineering has not authorized writes to this node."
                ),
            )
        # Range check
        if mapping.min_range is not None and value < mapping.min_range:
            return WriteResult(
                status=WriteStatus.BLOCKED_OUT_OF_RANGE,
                axion_tag=axion_tag,
                requested_value=value,
                error_message=(
                    f"Value {value} below min_range {mapping.min_range} for {axion_tag}"
                ),
            )
        if mapping.max_range is not None and value > mapping.max_range:
            return WriteResult(
                status=WriteStatus.BLOCKED_OUT_OF_RANGE,
                axion_tag=axion_tag,
                requested_value=value,
                error_message=(
                    f"Value {value} above max_range {mapping.max_range} for {axion_tag}"
                ),
            )

        if self._client is None:
            return WriteResult(
                status=WriteStatus.NOT_CONNECTED,
                axion_tag=axion_tag,
                requested_value=value,
                error_message="OPC-UA client not connected",
            )

        try:
            node = self._client.get_node(mapping.node_id)
            variant = ua.Variant(float(value), ua.VariantType.Double)
            await node.write_value(variant)
        except Exception as e:
            return WriteResult(
                status=WriteStatus.WRITE_ERROR,
                axion_tag=axion_tag,
                requested_value=value,
                error_message=f"OPC-UA write failed: {e}",
            )

        # Read-back verification
        await asyncio.sleep(self.readback_delay_s)
        try:
            readback = float(await node.read_value())
        except Exception as e:
            return WriteResult(
                status=WriteStatus.WRITE_ERROR,
                axion_tag=axion_tag,
                requested_value=value,
                error_message=f"OPC-UA readback failed: {e}",
            )

        denom = max(abs(value), 1e-9)
        if abs(readback - value) / denom > self.readback_tolerance:
            return WriteResult(
                status=WriteStatus.READBACK_MISMATCH,
                axion_tag=axion_tag,
                requested_value=value,
                readback_value=readback,
                error_message=(
                    f"Readback {readback} does not match requested {value} "
                    f"(tolerance {self.readback_tolerance*100:.0f}%)"
                ),
            )

        return WriteResult(
            status=WriteStatus.SUCCESS,
            axion_tag=axion_tag,
            requested_value=value,
            readback_value=readback,
        )
