"""
Axion AI — In-memory rate limiter
==================================

Token-bucket rate limiter keyed by (identity, scope). Designed for the
demo and small production deployments where:

  - There's a single API server process (no shared state across pods).
  - We want zero infra dependencies (no Redis).
  - Limits are coarse — per-user / per-IP / per-minute.

For multi-replica production a Redis-backed fixed-window counter would
be a drop-in replacement; this module's `RateLimiter` interface is the
contract.

Configuration via environment:
    AXION_RATE_LIMIT_PER_MIN  Requests per minute per identity. 0 disables.
                              Default: 120.
    AXION_RATE_LIMIT_BURST    Maximum burst size. Defaults to the per-min
                              limit (i.e. a bucket fills in 60s).

Identity resolution priority (used by the server middleware):
    1. authenticated user id (from JWT claims)         → "user:42"
    2. API key                                          → "key:<sha8>"
    3. client IP                                        → "ip:1.2.3.4"

The exempt-paths list keeps `/api/health` and `/api/metrics` always free
so health checks and Prometheus scrapes never get rate-limited.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


EXEMPT_PATHS = frozenset({
    "/api/health",
    "/api/metrics",
})


def per_min_from_env() -> int:
    return int(os.environ.get("AXION_RATE_LIMIT_PER_MIN", "120"))


def burst_from_env(per_min: int) -> int:
    raw = os.environ.get("AXION_RATE_LIMIT_BURST", "").strip()
    if not raw:
        return max(per_min, 1)
    return max(int(raw), 1)


@dataclass
class _Bucket:
    tokens:    float
    last_refill_ts: float


class RateLimiter:
    """Token-bucket limiter. Thread-safe.

    `per_min` tokens are added to a bucket of size `burst`; each request
    consumes 1. `allow(identity)` returns (allowed, retry_after_seconds)
    where retry_after estimates seconds until the next token if denied.
    """

    def __init__(self, per_min: int, burst: Optional[int] = None):
        if per_min < 0:
            raise ValueError("per_min must be >= 0")
        self.per_min = per_min
        self.burst = max(burst if burst is not None else per_min, 1)
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.per_min > 0

    @classmethod
    def from_env(cls) -> "RateLimiter":
        pm = per_min_from_env()
        return cls(per_min=pm, burst=burst_from_env(pm))

    def _refill(self, bucket: _Bucket, now: float) -> None:
        if self.per_min <= 0:
            return
        elapsed = max(0.0, now - bucket.last_refill_ts)
        tokens_per_sec = self.per_min / 60.0
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * tokens_per_sec)
        bucket.last_refill_ts = now

    def allow(self, identity: str) -> Tuple[bool, float]:
        """Try to consume one token for `identity`. Returns (allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), last_refill_ts=now)
                self._buckets[identity] = bucket
            self._refill(bucket, now)
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            # Time until 1 token regenerates
            tokens_per_sec = self.per_min / 60.0
            deficit = 1.0 - bucket.tokens
            retry_after = deficit / tokens_per_sec if tokens_per_sec > 0 else 1.0
            return False, retry_after

    def reset(self, identity: Optional[str] = None) -> None:
        """Clear the bucket for `identity`, or all buckets if None."""
        with self._lock:
            if identity is None:
                self._buckets.clear()
            else:
                self._buckets.pop(identity, None)


# ─────────────────────────────────────────────────────────────────────────────
# Identity helpers — server middleware calls these
# ─────────────────────────────────────────────────────────────────────────────

import hashlib


def short_hash(value: str, length: int = 8) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def resolve_identity(
    *,
    user_id: Optional[int],
    api_key: Optional[str],
    client_ip: Optional[str],
) -> str:
    """Pick the most specific identity available for rate limiting.

    A logged-in user gets their own bucket regardless of source IP, so a
    misbehaving NAT'd network doesn't lock out a legitimate user. Falls
    back to API key, then to client IP, then to the literal "anonymous"
    so callers always get a valid string.
    """
    if user_id is not None:
        return f"user:{int(user_id)}"
    if api_key:
        return f"key:{short_hash(api_key)}"
    if client_ip:
        return f"ip:{client_ip}"
    return "anonymous"
