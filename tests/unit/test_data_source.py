"""Tests for api/data_source.py + the source-switch endpoints."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from api.data_source import LIVE_COLUMNS, OpcuaBuffer
from integration.opcua_source import Sample


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample(ts: float = 1735689600.0, **values) -> Sample:
    """Build a Sample from a dict of axion_tag -> value."""
    s = Sample(timestamp=ts)
    s.values  = dict(values)
    s.quality = {k: 0 for k in values}
    return s


def _full_sample(ts: float, scale: float = 1.0) -> Sample:
    """A Sample with every canonical column populated."""
    return _sample(ts, **{
        col: scale * (i + 1)
        for i, col in enumerate(LIVE_COLUMNS[1:])   # skip "timestamp"
    })


# ─────────────────────────────────────────────────────────────────────────────
# OpcuaBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestOpcuaBuffer:
    def test_starts_empty(self):
        b = OpcuaBuffer()
        assert len(b) == 0
        assert b.to_dataframe().empty

    def test_append_increases_length(self):
        b = OpcuaBuffer()
        b.append(_full_sample(1.0))
        assert len(b) == 1

    def test_capacity_caps_size(self):
        b = OpcuaBuffer(capacity=3)
        for i in range(10):
            b.append(_full_sample(float(i)))
        assert len(b) == 3

    def test_capacity_keeps_most_recent(self):
        b = OpcuaBuffer(capacity=3)
        for i in range(10):
            b.append(_full_sample(float(i)))
        df = b.to_dataframe()
        # The 3 most recent samples are i=7,8,9 → timestamps 7..9
        ts_unix = [t.timestamp() for t in df["timestamp"]]
        assert ts_unix == [7.0, 8.0, 9.0]

    def test_to_dataframe_has_canonical_columns(self):
        b = OpcuaBuffer()
        b.append(_full_sample(1.0))
        df = b.to_dataframe()
        assert list(df.columns) == LIVE_COLUMNS

    def test_missing_tags_become_nan(self):
        b = OpcuaBuffer()
        # Only one tag is provided
        b.append(_sample(1.0, **{"cstr.T_R_C": 80.0}))
        df = b.to_dataframe()
        assert df.loc[0, "cstr.T_R_C"] == 80.0
        assert np.isnan(df.loc[0, "column.purity_B"])

    def test_unknown_tags_are_dropped(self):
        b = OpcuaBuffer()
        b.append(_sample(1.0, **{"cstr.T_R_C": 80.0, "not.a.tag": 999.0}))
        df = b.to_dataframe()
        assert "not.a.tag" not in df.columns

    def test_invalid_timestamp_falls_back_to_now(self):
        b = OpcuaBuffer()
        s = _full_sample(0)
        s.timestamp = "not-a-number"   # type: ignore[assignment]
        b.append(s)
        df = b.to_dataframe()
        # Should not raise — and timestamp must be parseable
        assert len(df) == 1
        assert pd.notna(df["timestamp"].iloc[0])

    def test_clear_empties_buffer(self):
        b = OpcuaBuffer()
        b.append(_full_sample(1.0))
        b.append(_full_sample(2.0))
        b.clear()
        assert len(b) == 0

    def test_append_many(self):
        b = OpcuaBuffer()
        b.append_many(_full_sample(float(i)) for i in range(5))
        assert len(b) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint: POST /api/data-source/select
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectDataSource:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_rejects_invalid_source(self, client):
        r = client.post("/api/data-source/select", json={"source": "magic"})
        assert r.status_code == 400

    def test_503_when_opcua_not_configured(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "opcua", None)
        r = client.post("/api/data-source/select", json={"source": "opcua"})
        assert r.status_code == 503

    def test_switch_to_opcua_sets_state(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "opcua", MagicMock(status=MagicMock(connected=True)))
        # populate the buffer with a few canonical samples
        server.state.opcua_buffer.clear()
        for i in range(40):
            server.state.opcua_buffer.append(_full_sample(float(i)))
        # Stub the analytics so we don't need a fitted engine
        monkeypatch.setattr(server.state, "ae",
                             MagicMock(run_sessions=MagicMock(return_value=[])))
        monkeypatch.setattr(server.state, "re",
                             MagicMock(generate=MagicMock(return_value=[])))
        r = client.post("/api/data-source/select", json={"source": "opcua"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["source"] == "opcua"
        assert server.state.data_source == "opcua"

    def test_switch_back_to_replay_calls_load_scenario(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "data_source", "opcua")
        # Stub load_scenario so we don't run the full pipeline
        called = {}
        def fake_load(name):
            called["name"] = name
            return MagicMock(scenario=name, process_data=pd.DataFrame({
                "timestamp": pd.date_range("2026-01-01", periods=2, freq="1min"),
            }), recommendations=[], decisions=[], sessions=[], performance=None)
        monkeypatch.setattr(server, "load_scenario", fake_load)
        r = client.post("/api/data-source/select", json={"source": "replay"})
        assert r.status_code == 200
        assert called["name"]   # load_scenario was invoked
        assert server.state.data_source == "replay"


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint: GET /api/data-source/status
# ─────────────────────────────────────────────────────────────────────────────

class TestDataSourceStatus:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_default_replay_when_unconfigured(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "data_source", "replay")
        monkeypatch.setattr(server.state, "opcua", None)
        body = client.get("/api/data-source/status").json()
        assert body["source"] == "replay"
        assert body["opcua_enabled"] is False
        assert body["opcua_connected"] is False

    def test_includes_buffer_count(self, client, monkeypatch):
        from api import server
        server.state.opcua_buffer.clear()
        for i in range(7):
            server.state.opcua_buffer.append(_full_sample(float(i)))
        body = client.get("/api/data-source/status").json()
        assert body["buffer_samples"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# /api/replay/status now exposes data_source
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayStatusExposesSource:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_data_source_field_present_no_run(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "run", None)
        monkeypatch.setattr(server.state, "data_source", "opcua")
        body = client.get("/api/replay/status").json()
        assert body["data_source"] == "opcua"

    def test_data_source_field_present_with_run(self, client, monkeypatch):
        from api import server
        run = MagicMock()
        run.scenario = "live"
        run.process_data = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
        })
        monkeypatch.setattr(server.state, "run", run)
        monkeypatch.setattr(server.state, "replay_idx", 1)
        monkeypatch.setattr(server.state, "data_source", "opcua")
        body = client.get("/api/replay/status").json()
        assert body["data_source"] == "opcua"
        assert body["scenario"] == "live"
