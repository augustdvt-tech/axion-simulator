"""Tests for integration/integration_service.py — env loading + lifecycle.

The tests do not require a real OPC-UA server. The OPCUASource is patched
where needed so we exercise the orchestration logic without networking.
"""

import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, ".")

from integration.integration_service import (
    IntegrationService,
    IntegrationStatus,
    _is_truthy,
    _override_from_env,
    load_tag_map_from_env,
)
from integration.tag_map import (
    PILOT_TAG_MAP, ServerConfig, SamplingConfig, TagMap, TagMapping,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_opcua_env(monkeypatch):
    for v in (
        "AXION_OPCUA_ENABLED", "AXION_OPCUA_TAG_MAP", "AXION_OPCUA_ENDPOINT",
        "AXION_OPCUA_USERNAME", "AXION_OPCUA_PASSWORD", "AXION_OPCUA_SECURITY",
        "AXION_OPCUA_CERT_PATH", "AXION_OPCUA_KEY_PATH", "AXION_OPCUA_TIME_NODE",
    ):
        monkeypatch.delenv(v, raising=False)


def _tiny_tag_map() -> TagMap:
    return TagMap(
        server=ServerConfig(endpoint="opc.tcp://127.0.0.1:4840"),
        sampling=SamplingConfig(interval_ms=1000),
        tags=[
            TagMapping("foo.bar", "ns=2;s=foo.bar"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# _is_truthy
# ─────────────────────────────────────────────────────────────────────────────

class TestIsTruthy:
    @pytest.mark.parametrize("v", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_truthy(self, v):
        assert _is_truthy(v) is True

    @pytest.mark.parametrize("v", ["", "false", "0", "no", "off", "maybe"])
    def test_falsy(self, v):
        assert _is_truthy(v) is False

    def test_strips_whitespace(self):
        assert _is_truthy("  true  ") is True


# ─────────────────────────────────────────────────────────────────────────────
# _override_from_env
# ─────────────────────────────────────────────────────────────────────────────

class TestOverrideFromEnv:
    def test_endpoint_override(self, monkeypatch):
        monkeypatch.setenv("AXION_OPCUA_ENDPOINT", "opc.tcp://other:4840")
        m = _override_from_env(_tiny_tag_map())
        assert m.server.endpoint == "opc.tcp://other:4840"

    def test_username_password_override(self, monkeypatch):
        monkeypatch.setenv("AXION_OPCUA_USERNAME", "axion-prod")
        monkeypatch.setenv("AXION_OPCUA_PASSWORD", "s3cret")
        m = _override_from_env(_tiny_tag_map())
        assert m.server.username == "axion-prod"
        assert m.server.password == "s3cret"

    def test_security_and_cert_paths(self, monkeypatch):
        monkeypatch.setenv("AXION_OPCUA_SECURITY",  "Basic256Sha256")
        monkeypatch.setenv("AXION_OPCUA_CERT_PATH", "/tmp/c.der")
        monkeypatch.setenv("AXION_OPCUA_KEY_PATH",  "/tmp/k.pem")
        m = _override_from_env(_tiny_tag_map())
        assert m.server.security_policy == "Basic256Sha256"
        assert m.server.cert_path == "/tmp/c.der"
        assert m.server.key_path  == "/tmp/k.pem"

    def test_unset_envs_leave_defaults(self):
        m = _override_from_env(_tiny_tag_map())
        assert m.server.endpoint == "opc.tcp://127.0.0.1:4840"
        assert m.server.username is None


# ─────────────────────────────────────────────────────────────────────────────
# load_tag_map_from_env
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadTagMapFromEnv:
    def test_falls_back_to_default(self):
        m = load_tag_map_from_env(default=_tiny_tag_map())
        assert m.tags[0].axion_tag == "foo.bar"

    def test_loads_from_file(self, tmp_path, monkeypatch):
        tag_file = tmp_path / "map.json"
        tag_file.write_text(json.dumps({
            "server":   {"endpoint": "opc.tcp://lab:4840"},
            "sampling": {"interval_ms": 500},
            "tags": [{
                "axion_tag": "x.y", "node_id": "ns=3;s=x.y",
            }],
        }))
        monkeypatch.setenv("AXION_OPCUA_TAG_MAP", str(tag_file))
        m = load_tag_map_from_env()
        assert m.server.endpoint == "opc.tcp://lab:4840"
        assert m.sampling.interval_ms == 500
        assert len(m.tags) == 1

    def test_env_overrides_file_endpoint(self, tmp_path, monkeypatch):
        tag_file = tmp_path / "map.json"
        tag_file.write_text(json.dumps({
            "server":   {"endpoint": "opc.tcp://lab:4840"},
            "sampling": {},
            "tags":     [],
        }))
        monkeypatch.setenv("AXION_OPCUA_TAG_MAP", str(tag_file))
        monkeypatch.setenv("AXION_OPCUA_ENDPOINT", "opc.tcp://prod:4840")
        m = load_tag_map_from_env()
        assert m.server.endpoint == "opc.tcp://prod:4840"


# ─────────────────────────────────────────────────────────────────────────────
# IntegrationStatus
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationStatus:
    def test_default_disabled(self):
        s = IntegrationStatus()
        d = s.to_dict()
        assert d["enabled"]   is False
        assert d["connected"] is False
        assert d["samples_received"] == 0

    def test_to_dict_keys(self):
        d = IntegrationStatus().to_dict()
        for key in ("enabled", "connected", "endpoint", "last_sample_ts",
                    "samples_received", "last_error", "n_tags", "started_at"):
            assert key in d


# ─────────────────────────────────────────────────────────────────────────────
# IntegrationService.from_env
# ─────────────────────────────────────────────────────────────────────────────

class TestFromEnv:
    def test_returns_none_when_disabled(self):
        assert IntegrationService.from_env() is None

    def test_returns_instance_when_enabled(self, monkeypatch):
        monkeypatch.setenv("AXION_OPCUA_ENABLED", "true")
        svc = IntegrationService.from_env()
        assert isinstance(svc, IntegrationService)
        assert svc.status.n_tags == len(PILOT_TAG_MAP.tags)
        assert svc.status.enabled is False   # not yet started


# ─────────────────────────────────────────────────────────────────────────────
# IntegrationService callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestServiceCallbacks:
    def test_handle_sample_updates_status(self):
        svc = IntegrationService(tag_map=_tiny_tag_map())
        from integration.opcua_source import Sample
        sample = Sample(timestamp=1234.5,
                        values={"foo.bar": 1.0}, quality={"foo.bar": 0})
        asyncio.run(svc._handle_sample(sample))
        assert svc.status.connected is True
        assert svc.status.samples_received == 1
        assert svc.status.last_sample_ts == 1234.5

    def test_handle_sample_invokes_user_callback(self):
        sink = []
        async def on_sample(s):
            sink.append(s)
        svc = IntegrationService(tag_map=_tiny_tag_map(), on_sample=on_sample)
        from integration.opcua_source import Sample
        asyncio.run(svc._handle_sample(Sample(timestamp=1.0)))
        assert len(sink) == 1

    def test_handle_event_connected_clears_error(self):
        svc = IntegrationService(tag_map=_tiny_tag_map())
        svc.status.last_error = "previous"
        asyncio.run(svc._handle_event("connected", {}))
        assert svc.status.connected  is True
        assert svc.status.last_error is None

    def test_handle_event_error_marks_disconnected(self):
        svc = IntegrationService(tag_map=_tiny_tag_map())
        svc.status.connected = True
        asyncio.run(svc._handle_event("error", {"error": "boom"}))
        assert svc.status.connected  is False
        assert svc.status.last_error == "boom"

    def test_user_callback_exception_is_captured(self):
        async def on_sample(_):
            raise RuntimeError("downstream broke")
        svc = IntegrationService(tag_map=_tiny_tag_map(), on_sample=on_sample)
        from integration.opcua_source import Sample
        asyncio.run(svc._handle_sample(Sample(timestamp=1.0)))
        # samples are still counted, error is captured
        assert svc.status.samples_received == 1
        assert "downstream broke" in (svc.status.last_error or "")


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle (mocked OPCUASource — no real network)
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_creates_task_and_marks_enabled(self, monkeypatch):
        async def runner():
            svc = IntegrationService(tag_map=_tiny_tag_map())
            with patch("integration.integration_service.OPCUASource") as MockSrc:
                instance = MockSrc.return_value
                # source.run never returns until stop
                async def fake_run():
                    while not getattr(instance, "_stopped", False):
                        await asyncio.sleep(0.01)
                instance.run = AsyncMock(side_effect=fake_run)
                instance.stop = MagicMock(
                    side_effect=lambda: setattr(instance, "_stopped", True),
                )
                await svc.start()
                assert svc.status.enabled is True
                assert svc._task is not None
                assert not svc._task.done()
                await svc.stop()
                assert svc.status.enabled is False
        asyncio.run(runner())

    def test_stop_is_safe_when_never_started(self):
        async def runner():
            svc = IntegrationService(tag_map=_tiny_tag_map())
            await svc.stop()   # should not raise
            assert svc.status.enabled is False
        asyncio.run(runner())


# ─────────────────────────────────────────────────────────────────────────────
# /api/integration/opcua/status endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_disabled_when_not_configured(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "opcua", None)
        body = client.get("/api/integration/opcua/status").json()
        assert body["enabled"] is False
        assert body["connected"] is False

    def test_returns_status_when_configured(self, client, monkeypatch):
        from api import server
        svc = IntegrationService(tag_map=_tiny_tag_map())
        svc.status.enabled = True
        svc.status.connected = True
        svc.status.samples_received = 42
        monkeypatch.setattr(server.state, "opcua", svc)
        body = client.get("/api/integration/opcua/status").json()
        assert body["enabled"] is True
        assert body["connected"] is True
        assert body["samples_received"] == 42
        assert body["endpoint"] == "opc.tcp://127.0.0.1:4840"
