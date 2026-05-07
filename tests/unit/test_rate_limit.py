"""Tests for api/rate_limit.py."""

import sys
import time

import pytest

sys.path.insert(0, ".")

from api.rate_limit import (
    EXEMPT_PATHS, RateLimiter, resolve_identity, short_hash,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in ("AXION_RATE_LIMIT_PER_MIN", "AXION_RATE_LIMIT_BURST"):
        monkeypatch.delenv(v, raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveIdentity:
    def test_user_id_wins(self):
        assert resolve_identity(user_id=42, api_key="k", client_ip="1.2.3.4") \
            == "user:42"

    def test_api_key_when_no_user(self):
        ident = resolve_identity(user_id=None, api_key="secret", client_ip="1.2.3.4")
        assert ident.startswith("key:")
        assert len(ident) > len("key:")

    def test_ip_fallback(self):
        assert resolve_identity(user_id=None, api_key=None, client_ip="9.9.9.9") \
            == "ip:9.9.9.9"

    def test_anonymous_when_nothing(self):
        assert resolve_identity(user_id=None, api_key=None, client_ip=None) \
            == "anonymous"

    def test_short_hash_deterministic(self):
        assert short_hash("foo") == short_hash("foo")
        assert short_hash("foo") != short_hash("bar")


class TestExemptPaths:
    def test_health_exempt(self):
        assert "/api/health" in EXEMPT_PATHS

    def test_metrics_exempt(self):
        assert "/api/metrics" in EXEMPT_PATHS


# ─────────────────────────────────────────────────────────────────────────────
# RateLimiter
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_disabled_when_per_min_zero(self):
        rl = RateLimiter(per_min=0)
        assert rl.enabled is False
        for _ in range(1000):
            ok, _ = rl.allow("any")
            assert ok is True

    def test_negative_per_min_raises(self):
        with pytest.raises(ValueError):
            RateLimiter(per_min=-1)

    def test_first_burst_allowed(self):
        rl = RateLimiter(per_min=60, burst=5)
        for _ in range(5):
            assert rl.allow("alice")[0] is True

    def test_over_burst_denied(self):
        rl = RateLimiter(per_min=60, burst=3)
        for _ in range(3):
            assert rl.allow("alice")[0] is True
        ok, retry = rl.allow("alice")
        assert ok is False
        assert retry > 0

    def test_independent_identities(self):
        rl = RateLimiter(per_min=60, burst=2)
        assert rl.allow("alice")[0] is True
        assert rl.allow("alice")[0] is True
        # Alice exhausted, but bob is fresh
        assert rl.allow("alice")[0] is False
        assert rl.allow("bob")[0]   is True

    def test_tokens_replenish_over_time(self, monkeypatch):
        rl = RateLimiter(per_min=60, burst=1)   # 1 token/sec
        # Simulate clock advance via monkeypatching time.monotonic
        t = [1000.0]
        monkeypatch.setattr("api.rate_limit.time.monotonic", lambda: t[0])
        assert rl.allow("alice")[0] is True
        assert rl.allow("alice")[0] is False
        # Advance 2 seconds → at least 1 token regenerated
        t[0] += 2.0
        assert rl.allow("alice")[0] is True

    def test_retry_after_estimates_seconds(self):
        rl = RateLimiter(per_min=60, burst=1)   # 1 token/sec
        rl.allow("alice")
        ok, retry = rl.allow("alice")
        assert ok is False
        # Should be approximately 1 second to wait for the next token
        assert 0.5 <= retry <= 1.5

    def test_reset_specific_identity(self):
        rl = RateLimiter(per_min=60, burst=1)
        rl.allow("alice")
        assert rl.allow("alice")[0] is False
        rl.reset("alice")
        assert rl.allow("alice")[0] is True

    def test_reset_all(self):
        rl = RateLimiter(per_min=60, burst=1)
        rl.allow("a")
        rl.allow("b")
        assert rl.allow("a")[0] is False
        assert rl.allow("b")[0] is False
        rl.reset()
        assert rl.allow("a")[0] is True
        assert rl.allow("b")[0] is True


class TestFromEnv:
    def test_default_per_min_120(self):
        rl = RateLimiter.from_env()
        assert rl.per_min == 120

    def test_picks_up_per_min(self, monkeypatch):
        monkeypatch.setenv("AXION_RATE_LIMIT_PER_MIN", "30")
        rl = RateLimiter.from_env()
        assert rl.per_min == 30

    def test_burst_defaults_to_per_min(self, monkeypatch):
        monkeypatch.setenv("AXION_RATE_LIMIT_PER_MIN", "60")
        rl = RateLimiter.from_env()
        assert rl.burst == 60

    def test_burst_override(self, monkeypatch):
        monkeypatch.setenv("AXION_RATE_LIMIT_PER_MIN", "60")
        monkeypatch.setenv("AXION_RATE_LIMIT_BURST", "10")
        rl = RateLimiter.from_env()
        assert rl.burst == 10

    def test_zero_per_min_disabled(self, monkeypatch):
        monkeypatch.setenv("AXION_RATE_LIMIT_PER_MIN", "0")
        rl = RateLimiter.from_env()
        assert rl.enabled is False
