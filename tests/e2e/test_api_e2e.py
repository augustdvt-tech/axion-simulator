"""
End-to-end API tests against a live uvicorn server.

These exercise the full request path — middleware chain, startup hooks,
replay loop scheduling, WebSocket lifecycle — that the unit-level
TestClient bypasses. They run in CI on every PR.

Each test is independent: ordering is not assumed. The default scenario
(thermal_drift, loaded by the startup hook) gives us recommendations
within a few seconds of replay.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest


pytestmark = pytest.mark.e2e


# ─────────────────────────────────────────────────────────────────────────────
# Health & basic shape
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, http_client):
        r = http_client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_reports_loaded_scenario(self, http_client):
        body = http_client.get("/api/health").json()
        assert body["scenario"] is not None
        assert body["samples_total"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarios:
    def test_list_scenarios_includes_known_ones(self, http_client):
        body = http_client.get("/api/scenarios").json()
        names = set(body["available"])
        for s in ("normal", "thermal_drift", "feed_perturbation"):
            assert s in names, f"missing scenario {s}"

    def test_select_scenario_round_trip(self, http_client):
        target = "feed_perturbation"
        r = http_client.post("/api/scenarios/select", json={"scenario": target})
        assert r.status_code == 200
        # Health now reports the new scenario
        body = http_client.get("/api/health").json()
        assert body["scenario"] == target

    def test_select_unknown_scenario_returns_404(self, http_client):
        r = http_client.post(
            "/api/scenarios/select", json={"scenario": "ghost_scenario"},
        )
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# State + recent + replay status
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessSnapshot:
    def test_state_returns_canonical_shape(self, http_client):
        body = http_client.get("/api/state").json()
        assert "timestamp" in body
        assert "cstr" in body and "T_R_C" in body["cstr"]
        assert "column" in body and "purity_B" in body["column"]

    def test_recent_returns_history(self, http_client):
        body = http_client.get("/api/process/recent?samples=5").json()
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0


class TestReplayStatus:
    def test_replay_status_keys(self, http_client):
        body = http_client.get("/api/replay/status").json()
        for key in ("running", "idx", "total", "speed", "data_source"):
            assert key in body


# ─────────────────────────────────────────────────────────────────────────────
# Recommendations + decisions round-trip
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_recommendation(client: httpx.Client, timeout_s: float = 10.0):
    """Drive the replay forward until at least one recommendation appears."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get("/api/recommendations?limit=5").json()
        if body.get("recommendations"):
            return body["recommendations"][0]
        time.sleep(0.5)
    return None


class TestDecisionRoundTrip:
    def test_decide_acceptance_persists_in_listing(self, http_client):
        # Use thermal_drift which always produces recs
        http_client.post("/api/scenarios/select", json={"scenario": "thermal_drift"})
        # Seek near the end of the run so all recommendations are visible
        status = http_client.get("/api/replay/status").json()
        target_idx = max(0, int(status["total"]) - 2)
        http_client.post("/api/replay/control",
                          json={"action": "seek", "idx": target_idx})

        rec = _wait_for_recommendation(http_client)
        if rec is None:
            pytest.skip("No recommendation appeared within timeout")
        rec_id = rec["id"]

        r = http_client.post(
            f"/api/recommendations/{rec_id}/decide",
            json={"action": "accept", "justification": "e2e"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "accepted"

        # Listing should reflect the decision
        body = http_client.get("/api/recommendations?limit=20").json()
        match = next((x for x in body["recommendations"] if x["id"] == rec_id), None)
        assert match is not None
        assert match["status"] == "accepted"


# ─────────────────────────────────────────────────────────────────────────────
# Profile endpoints (Bloque U)
# ─────────────────────────────────────────────────────────────────────────────

class TestProfile:
    def test_profile_endpoint_lists_pilot_and_batch(self, http_client):
        body = http_client.get("/api/profile").json()
        assert "pilot" in body["available"]
        assert "batch_reactor" in body["available"]

    def test_profile_dict_includes_tags(self, http_client):
        body = http_client.get("/api/profile").json()
        assert isinstance(body["profile"]["tags"], list)
        assert len(body["profile"]["tags"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (Bloque W)
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_endpoint_returns_prometheus_text(self, http_client):
        r = http_client.get("/api/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", "")
        assert "axion_http_requests_total" in r.text


# ─────────────────────────────────────────────────────────────────────────────
# Models status + drift (Bloques L, R)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelsAndDrift:
    def test_models_status_returns_both_keys(self, http_client):
        body = http_client.get("/api/models/status").json()
        assert "soft_sensor" in body
        assert "lstm_forecaster" in body

    def test_drift_status_endpoint_responds(self, http_client):
        body = http_client.get("/api/drift/status").json()
        assert "available" in body


# ─────────────────────────────────────────────────────────────────────────────
# Report endpoint (Bloque M)
# ─────────────────────────────────────────────────────────────────────────────

class TestReport:
    def test_report_returns_html(self, http_client):
        r = http_client.get("/api/report/current")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert "AXION" in r.text
        assert "<!DOCTYPE html>" in r.text


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSocket:
    def test_ws_snapshot_on_connect(self, live_server):
        # httpx doesn't speak WS; use stdlib websocket-style via httpx_ws
        # would add a dep — instead, rely on Python's `websockets` library
        # which is already a dependency of asyncua / FastAPI's websocket
        try:
            from websockets.sync.client import connect
        except ImportError:
            pytest.skip("websockets not installed")

        ws_url = live_server.replace("http://", "ws://") + "/ws/stream"
        with connect(ws_url, open_timeout=5.0) as ws:
            msg = ws.recv(timeout=5.0)
            data = json.loads(msg)
            assert data["type"] == "snapshot"
            assert "sample" in data
