"""Tests for the RBAC middleware (viewer / operator / manager roles)."""

import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, ".")


@pytest.fixture(autouse=True)
def _clear_keys(monkeypatch):
    """Wipe all RBAC env vars before each test for hermetic state."""
    for var in (
        "AXION_API_KEY",
        "AXION_API_KEY_VIEWER",
        "AXION_API_KEY_OPERATOR",
        "AXION_API_KEY_MANAGER",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def client():
    from api.server import app
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRoleHelpers:
    def test_required_role_get_is_viewer(self):
        from api.server import _required_role
        assert _required_role("GET", "/api/state") == "viewer"
        assert _required_role("GET", "/api/recommendations") == "viewer"

    def test_required_role_decide_is_operator(self):
        from api.server import _required_role
        assert _required_role(
            "POST", "/api/recommendations/REC-001/decide"
        ) == "operator"

    def test_required_role_replay_control_is_operator(self):
        from api.server import _required_role
        assert _required_role("POST", "/api/replay/control") == "operator"

    def test_required_role_optimization_predict_is_operator(self):
        from api.server import _required_role
        assert _required_role("POST", "/api/optimization/predict") == "operator"

    def test_required_role_scenario_select_is_manager(self):
        from api.server import _required_role
        assert _required_role("POST", "/api/scenarios/select") == "manager"

    def test_role_satisfies_hierarchy(self):
        from api.server import _role_satisfies
        # manager > operator > viewer
        assert _role_satisfies("manager", "viewer")
        assert _role_satisfies("manager", "operator")
        assert _role_satisfies("manager", "manager")
        assert _role_satisfies("operator", "viewer")
        assert _role_satisfies("operator", "operator")
        assert not _role_satisfies("operator", "manager")
        assert _role_satisfies("viewer", "viewer")
        assert not _role_satisfies("viewer", "operator")
        assert not _role_satisfies("viewer", "manager")

    def test_load_role_keys_empty_when_unset(self, monkeypatch):
        from api.server import _load_role_keys
        assert _load_role_keys() == {}

    def test_load_role_keys_legacy_maps_to_manager(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY", "legacy")
        from api.server import _load_role_keys
        assert _load_role_keys() == {"legacy": "manager"}

    def test_load_role_keys_three_roles(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY_VIEWER", "v1")
        monkeypatch.setenv("AXION_API_KEY_OPERATOR", "o1")
        monkeypatch.setenv("AXION_API_KEY_MANAGER", "m1")
        from api.server import _load_role_keys
        keys = _load_role_keys()
        assert keys["v1"] == "viewer"
        assert keys["o1"] == "operator"
        assert keys["m1"] == "manager"


# ─────────────────────────────────────────────────────────────────────────────
# RBAC disabled (no env vars)
# ─────────────────────────────────────────────────────────────────────────────

class TestRBACDisabled:
    def test_get_works_without_key(self, client):
        r = client.get("/api/scenarios")
        assert r.status_code != 401
        assert r.status_code != 403

    def test_post_works_without_key(self, client):
        r = client.post("/api/replay/control", json={"action": "pause"})
        assert r.status_code != 401
        assert r.status_code != 403


# ─────────────────────────────────────────────────────────────────────────────
# RBAC enabled — viewer role
# ─────────────────────────────────────────────────────────────────────────────

class TestViewerRole:
    @pytest.fixture
    def viewer_client(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY_VIEWER", "v-key")
        monkeypatch.setenv("AXION_API_KEY_OPERATOR", "o-key")
        monkeypatch.setenv("AXION_API_KEY_MANAGER", "m-key")
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_viewer_can_get_health(self, viewer_client):
        # health is always public, no key required
        r = viewer_client.get("/api/health")
        assert r.status_code == 200

    def test_viewer_can_get_scenarios(self, viewer_client):
        r = viewer_client.get("/api/scenarios", headers={"X-API-Key": "v-key"})
        assert r.status_code != 401
        assert r.status_code != 403

    def test_viewer_cannot_decide(self, viewer_client):
        r = viewer_client.post(
            "/api/recommendations/REC-001/decide",
            headers={"X-API-Key": "v-key"},
            json={"action": "accept"},
        )
        assert r.status_code == 403

    def test_viewer_cannot_change_scenario(self, viewer_client):
        r = viewer_client.post(
            "/api/scenarios/select",
            headers={"X-API-Key": "v-key"},
            json={"scenario": "thermal_drift"},
        )
        assert r.status_code == 403

    def test_viewer_cannot_control_replay(self, viewer_client):
        r = viewer_client.post(
            "/api/replay/control",
            headers={"X-API-Key": "v-key"},
            json={"action": "pause"},
        )
        assert r.status_code == 403

    def test_403_response_includes_required_role(self, viewer_client):
        r = viewer_client.post(
            "/api/scenarios/select",
            headers={"X-API-Key": "v-key"},
            json={"scenario": "thermal_drift"},
        )
        body = r.json()
        assert body.get("required_role") == "manager"
        assert body.get("actual_role") == "viewer"


# ─────────────────────────────────────────────────────────────────────────────
# RBAC enabled — operator role
# ─────────────────────────────────────────────────────────────────────────────

class TestOperatorRole:
    @pytest.fixture
    def op_client(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY_VIEWER", "v-key")
        monkeypatch.setenv("AXION_API_KEY_OPERATOR", "o-key")
        monkeypatch.setenv("AXION_API_KEY_MANAGER", "m-key")
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_operator_can_read(self, op_client):
        r = op_client.get("/api/scenarios", headers={"X-API-Key": "o-key"})
        assert r.status_code != 401
        assert r.status_code != 403

    def test_operator_can_control_replay(self, op_client):
        r = op_client.post(
            "/api/replay/control",
            headers={"X-API-Key": "o-key"},
            json={"action": "pause"},
        )
        # 200 if scenario loaded, 404 if not — never 403
        assert r.status_code != 403

    def test_operator_can_decide(self, op_client):
        r = op_client.post(
            "/api/recommendations/REC-001/decide",
            headers={"X-API-Key": "o-key"},
            json={"action": "accept"},
        )
        # auth passes — endpoint returns 404 (no scenario) but not 403
        assert r.status_code != 403

    def test_operator_cannot_change_scenario(self, op_client):
        r = op_client.post(
            "/api/scenarios/select",
            headers={"X-API-Key": "o-key"},
            json={"scenario": "thermal_drift"},
        )
        assert r.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# RBAC enabled — manager role
# ─────────────────────────────────────────────────────────────────────────────

class TestManagerRole:
    @pytest.fixture
    def mgr_client(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY_VIEWER", "v-key")
        monkeypatch.setenv("AXION_API_KEY_OPERATOR", "o-key")
        monkeypatch.setenv("AXION_API_KEY_MANAGER", "m-key")
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_manager_can_read(self, mgr_client):
        r = mgr_client.get("/api/scenarios", headers={"X-API-Key": "m-key"})
        assert r.status_code != 403

    def test_manager_can_change_scenario(self, mgr_client):
        r = mgr_client.post(
            "/api/scenarios/select",
            headers={"X-API-Key": "m-key"},
            json={"scenario": "thermal_drift"},
        )
        assert r.status_code != 403

    def test_manager_can_decide(self, mgr_client):
        r = mgr_client.post(
            "/api/recommendations/REC-001/decide",
            headers={"X-API-Key": "m-key"},
            json={"action": "accept"},
        )
        assert r.status_code != 403


# ─────────────────────────────────────────────────────────────────────────────
# Invalid / missing keys with RBAC enabled
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidKey:
    @pytest.fixture
    def auth_client(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY_VIEWER", "v-key")
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_missing_key_returns_401(self, auth_client):
        r = auth_client.get("/api/scenarios")
        assert r.status_code == 401

    def test_unknown_key_returns_401(self, auth_client):
        r = auth_client.get(
            "/api/scenarios", headers={"X-API-Key": "ghost-key"}
        )
        assert r.status_code == 401

    def test_health_still_public_with_rbac_on(self, auth_client):
        r = auth_client.get("/api/health")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Legacy AXION_API_KEY (backward compat) maps to manager
# ─────────────────────────────────────────────────────────────────────────────

class TestLegacyKey:
    @pytest.fixture
    def legacy_client(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY", "legacy-key")
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_legacy_key_can_read(self, legacy_client):
        r = legacy_client.get(
            "/api/scenarios", headers={"X-API-Key": "legacy-key"}
        )
        assert r.status_code != 403

    def test_legacy_key_can_change_scenario(self, legacy_client):
        r = legacy_client.post(
            "/api/scenarios/select",
            headers={"X-API-Key": "legacy-key"},
            json={"scenario": "thermal_drift"},
        )
        assert r.status_code != 403


# ─────────────────────────────────────────────────────────────────────────────
# Query param fallback (used by WebSocket clients)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryParamFallback:
    @pytest.fixture
    def auth_client(self, monkeypatch):
        monkeypatch.setenv("AXION_API_KEY_VIEWER", "v-key")
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_query_param_accepted(self, auth_client):
        r = auth_client.get("/api/scenarios", params={"api_key": "v-key"})
        assert r.status_code != 401

    def test_wrong_query_param_rejected(self, auth_client):
        r = auth_client.get("/api/scenarios", params={"api_key": "wrong"})
        assert r.status_code == 401
