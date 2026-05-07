"""Tests for api/webhooks.py — WebhookNotifier + helpers."""

import json
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from api.webhooks import (
    WebhookNotifier,
    _format_payload,
    _meets_threshold,
)


def _rec(rec_id="REC-1", urgency="critical", rule="R01", action="Reduce setpoint"):
    return {
        "id":         rec_id,
        "timestamp":  "2026-01-01T00:00:00",
        "urgency":    urgency,
        "rule_fired": rule,
        "diagnosis":  "thermal drift detected",
        "action":     action,
    }


# ─────────────────────────────────────────────────────────────────────────────
# _meets_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestMeetsThreshold:
    def test_critical_meets_critical(self):
        assert _meets_threshold("critical", "critical")

    def test_high_does_not_meet_critical(self):
        assert not _meets_threshold("high", "critical")

    def test_critical_meets_high(self):
        assert _meets_threshold("critical", "high")

    def test_unknown_urgency_does_not_meet(self):
        assert not _meets_threshold("unknown", "critical")

    def test_low_meets_low(self):
        assert _meets_threshold("low", "low")


# ─────────────────────────────────────────────────────────────────────────────
# _format_payload
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatPayload:
    def test_axion_format_structure(self):
        body = _format_payload(_rec(), "thermal_drift", "axion")
        assert body["event"] == "recommendation"
        assert body["scenario"] == "thermal_drift"
        assert body["rec_id"] == "REC-1"
        assert body["urgency"] == "critical"

    def test_slack_format_has_text_key(self):
        body = _format_payload(_rec(), "thermal_drift", "slack")
        assert "text" in body
        assert isinstance(body["text"], str)

    def test_slack_format_includes_scenario(self):
        body = _format_payload(_rec(), "thermal_drift", "slack")
        assert "thermal_drift" in body["text"]

    def test_slack_format_includes_urgency_uppercase(self):
        body = _format_payload(_rec(urgency="critical"), "s", "slack")
        assert "CRITICAL" in body["text"]


# ─────────────────────────────────────────────────────────────────────────────
# WebhookNotifier behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestNotifierEnabled:
    def test_disabled_when_no_url(self):
        n = WebhookNotifier(url=None)
        assert n.enabled is False

    def test_enabled_when_url_set(self):
        n = WebhookNotifier(url="http://example.com/hook")
        assert n.enabled is True


class TestShouldFire:
    def test_disabled_notifier_never_fires(self):
        n = WebhookNotifier(url=None, threshold="critical")
        assert not n.should_fire(_rec())

    def test_critical_fires_at_critical_threshold(self):
        n = WebhookNotifier(url="http://x", threshold="critical")
        assert n.should_fire(_rec(urgency="critical"))

    def test_high_does_not_fire_at_critical_threshold(self):
        n = WebhookNotifier(url="http://x", threshold="critical")
        assert not n.should_fire(_rec(urgency="high"))

    def test_high_fires_at_high_threshold(self):
        n = WebhookNotifier(url="http://x", threshold="high")
        assert n.should_fire(_rec(urgency="high"))

    def test_dedupe_blocks_second_call(self):
        n = WebhookNotifier(url="http://x", threshold="critical")
        rec = _rec(rec_id="REC-A")
        assert n.should_fire(rec)
        n._fired.add("REC-A")
        assert not n.should_fire(rec)


class TestNotify:
    def test_notify_disabled_returns_false(self):
        n = WebhookNotifier(url=None)
        assert n.notify(_rec(), "scn") is False

    def test_notify_below_threshold_returns_false(self):
        n = WebhookNotifier(url="http://x", threshold="critical")
        assert n.notify(_rec(urgency="medium"), "scn") is False

    def test_notify_fires_synchronously(self):
        n = WebhookNotifier(url="http://x", threshold="critical", async_post=False)
        with patch("api.webhooks._post", return_value=True) as mock_post:
            ok = n.notify(_rec(), "scn")
        assert ok is True
        mock_post.assert_called_once()

    def test_notify_dedupe_second_call_returns_false(self):
        n = WebhookNotifier(url="http://x", threshold="critical", async_post=False)
        with patch("api.webhooks._post", return_value=True):
            assert n.notify(_rec(rec_id="REC-1"), "s") is True
            assert n.notify(_rec(rec_id="REC-1"), "s") is False

    def test_notify_different_ids_both_fire(self):
        n = WebhookNotifier(url="http://x", threshold="critical", async_post=False)
        with patch("api.webhooks._post", return_value=True) as mock_post:
            n.notify(_rec(rec_id="REC-1"), "s")
            n.notify(_rec(rec_id="REC-2"), "s")
        assert mock_post.call_count == 2

    def test_notify_returns_true_when_async(self):
        # async_post=True returns True immediately after spawning the thread
        n = WebhookNotifier(url="http://x", threshold="critical", async_post=True)
        with patch("api.webhooks._post", return_value=True):
            assert n.notify(_rec(), "s") is True

    def test_notify_passes_correct_url_and_body(self):
        n = WebhookNotifier(url="http://example.com/hook", threshold="critical",
                             async_post=False, fmt="axion")
        with patch("api.webhooks._post", return_value=True) as mock_post:
            n.notify(_rec(), "thermal_drift")
        args, _ = mock_post.call_args
        assert args[0] == "http://example.com/hook"
        assert args[1]["scenario"] == "thermal_drift"

    def test_notify_post_failure_does_not_raise(self):
        n = WebhookNotifier(url="http://x", threshold="critical", async_post=False)
        with patch("api.webhooks._post", return_value=False):
            # Should not raise even when POST fails
            assert n.notify(_rec(), "s") is False


class TestReset:
    def test_reset_clears_fired_set(self):
        n = WebhookNotifier(url="http://x", threshold="critical", async_post=False)
        with patch("api.webhooks._post", return_value=True):
            n.notify(_rec(rec_id="REC-1"), "s")
        assert "REC-1" in n._fired
        n.reset()
        assert n._fired == set()

    def test_after_reset_same_rec_fires_again(self):
        n = WebhookNotifier(url="http://x", threshold="critical", async_post=False)
        with patch("api.webhooks._post", return_value=True) as mock_post:
            n.notify(_rec(rec_id="REC-1"), "s")
            n.reset()
            n.notify(_rec(rec_id="REC-1"), "s")
        assert mock_post.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# from_env
# ─────────────────────────────────────────────────────────────────────────────

class TestFromEnv:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        for v in ("AXION_WEBHOOK_URL", "AXION_WEBHOOK_URGENCY",
                  "AXION_WEBHOOK_TIMEOUT", "AXION_WEBHOOK_FORMAT"):
            monkeypatch.delenv(v, raising=False)

    def test_defaults_when_unset(self):
        n = WebhookNotifier.from_env()
        assert n.enabled is False
        assert n.threshold == "critical"
        assert n.fmt == "axion"

    def test_picks_up_url(self, monkeypatch):
        monkeypatch.setenv("AXION_WEBHOOK_URL", "http://hook.example.com/x")
        n = WebhookNotifier.from_env()
        assert n.enabled is True
        assert n.url == "http://hook.example.com/x"

    def test_picks_up_threshold(self, monkeypatch):
        monkeypatch.setenv("AXION_WEBHOOK_URL", "http://x")
        monkeypatch.setenv("AXION_WEBHOOK_URGENCY", "high")
        n = WebhookNotifier.from_env()
        assert n.threshold == "high"

    def test_picks_up_format(self, monkeypatch):
        monkeypatch.setenv("AXION_WEBHOOK_URL", "http://x")
        monkeypatch.setenv("AXION_WEBHOOK_FORMAT", "slack")
        n = WebhookNotifier.from_env()
        assert n.fmt == "slack"

    def test_picks_up_timeout(self, monkeypatch):
        monkeypatch.setenv("AXION_WEBHOOK_URL", "http://x")
        monkeypatch.setenv("AXION_WEBHOOK_TIMEOUT", "12.5")
        n = WebhookNotifier.from_env()
        assert n.timeout == 12.5


# ─────────────────────────────────────────────────────────────────────────────
# /api/webhook/status endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestWebhookStatusEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_status_returns_200(self, client):
        r = client.get("/api/webhook/status")
        assert r.status_code == 200

    def test_status_shape(self, client):
        body = client.get("/api/webhook/status").json()
        for key in ("enabled", "threshold", "format", "n_fired"):
            assert key in body

    def test_status_disabled_by_default(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "webhook",
                            WebhookNotifier(url=None))
        body = client.get("/api/webhook/status").json()
        assert body["enabled"] is False

    def test_status_enabled_when_url_set(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "webhook",
                            WebhookNotifier(url="http://x", threshold="high"))
        body = client.get("/api/webhook/status").json()
        assert body["enabled"] is True
        assert body["threshold"] == "high"
