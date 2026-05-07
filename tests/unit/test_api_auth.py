"""Tests for api/server.py auth middleware (X-API-Key)."""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _clear_api_key(monkeypatch):
    """Ensure AXION_API_KEY is unset before each test."""
    monkeypatch.delenv("AXION_API_KEY", raising=False)


@pytest.fixture
def client():
    import sys
    sys.path.insert(0, ".")
    from api.server import app
    return TestClient(app, raise_server_exceptions=False)


class TestAuthDisabled:
    def test_health_always_public(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_state_accessible_without_key_when_auth_disabled(self, client):
        r = client.get("/api/state")
        # 200 or 404 (no scenario loaded in unit test context) — never 401
        assert r.status_code != 401

    def test_scenarios_accessible_without_key_when_auth_disabled(self, client):
        r = client.get("/api/scenarios")
        assert r.status_code != 401


class TestAuthEnabled:
    @pytest.fixture
    def auth_client(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY", "test-secret-key")
        import sys
        sys.path.insert(0, ".")
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_health_still_public_with_auth_enabled(self, auth_client):
        r = auth_client.get("/api/health")
        assert r.status_code == 200

    def test_protected_endpoint_rejects_missing_key(self, auth_client):
        r = auth_client.get("/api/scenarios")
        assert r.status_code == 401

    def test_protected_endpoint_rejects_wrong_key(self, auth_client):
        r = auth_client.get("/api/scenarios",
                            headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_protected_endpoint_accepts_correct_key(self, auth_client):
        r = auth_client.get("/api/scenarios",
                            headers={"X-API-Key": "test-secret-key"})
        assert r.status_code != 401

    def test_401_response_has_detail(self, auth_client):
        r = auth_client.get("/api/state")
        assert r.status_code == 401
        body = r.json()
        assert "detail" in body

    def test_query_param_accepted_for_websocket_path(self, auth_client):
        # WebSocket auth via ?api_key= (checked at HTTP upgrade level)
        # TestClient can't do real WS, so we verify the HTTP path check works
        r = auth_client.get("/api/scenarios",
                            params={"api_key": "test-secret-key"})
        assert r.status_code != 401
