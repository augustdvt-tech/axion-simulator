"""Tests for api/auth.py + the /api/auth/* endpoints (Bloque V)."""

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, ".")

from api.auth import (
    AuthError, InvalidTokenError, TokenExpiredError, TokenTypeMismatchError,
    decode_token, encode_token, extract_bearer_token, hash_password,
    issue_token_pair, verify_password,
)


SECRET = "test-secret-do-not-use-in-prod"


# ─────────────────────────────────────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_returns_bcrypt_string(self):
        h = hash_password("hunter2")
        assert h.startswith(("$2a$", "$2b$", "$2y$"))

    def test_verify_correct_password(self):
        h = hash_password("hunter2")
        assert verify_password("hunter2", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("hunter2")
        assert verify_password("wrong", h) is False

    def test_verify_empty_inputs(self):
        assert verify_password("", "")    is False
        assert verify_password("x", "")   is False
        assert verify_password("", "x")   is False

    def test_verify_malformed_hash_does_not_raise(self):
        assert verify_password("hunter2", "not-a-bcrypt-hash") is False

    def test_hash_empty_password_raises(self):
        with pytest.raises(ValueError):
            hash_password("")


# ─────────────────────────────────────────────────────────────────────────────
# Token encode / decode
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenEncodeDecode:
    def test_round_trip(self):
        token = encode_token(secret=SECRET, sub="a@b.com", uid=42,
                              role="manager", token_type="access",
                              ttl_seconds=60)
        claims = decode_token(token, SECRET, expected_type="access")
        assert claims.sub == "a@b.com"
        assert claims.uid == 42
        assert claims.role == "manager"
        assert claims.type == "access"

    def test_unknown_token_type_raises(self):
        with pytest.raises(ValueError):
            encode_token(secret=SECRET, sub="x", uid=1, role="viewer",
                          token_type="weird", ttl_seconds=60)

    def test_wrong_secret_rejected(self):
        token = encode_token(secret=SECRET, sub="x", uid=1, role="viewer",
                              token_type="access", ttl_seconds=60)
        with pytest.raises(InvalidTokenError):
            decode_token(token, "different-secret", expected_type="access")

    def test_expired_token_rejected(self):
        past = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())
        token = encode_token(secret=SECRET, sub="x", uid=1, role="viewer",
                              token_type="access", ttl_seconds=60,
                              issued_at=past - 60)
        with pytest.raises(TokenExpiredError):
            decode_token(token, SECRET, expected_type="access")

    def test_token_type_mismatch_rejected(self):
        access = encode_token(secret=SECRET, sub="x", uid=1, role="viewer",
                                token_type="access", ttl_seconds=60)
        with pytest.raises(TokenTypeMismatchError):
            decode_token(access, SECRET, expected_type="refresh")

    def test_no_expected_type_skips_check(self):
        token = encode_token(secret=SECRET, sub="x", uid=1, role="viewer",
                              token_type="access", ttl_seconds=60)
        claims = decode_token(token, SECRET, expected_type=None)
        assert claims.type == "access"

    def test_missing_claim_rejected(self):
        # craft a JWT with PyJWT directly that's missing 'role'
        import jwt
        payload = {"sub": "x", "uid": 1, "type": "access",
                   "iat": 0, "exp": 9999999999}
        bad = jwt.encode(payload, SECRET, algorithm="HS256")
        with pytest.raises(InvalidTokenError):
            decode_token(bad, SECRET)


class TestIssueTokenPair:
    def test_returns_distinct_access_and_refresh(self):
        a, r, _ = issue_token_pair(secret=SECRET, sub="x@y", uid=1,
                                     role="viewer")
        assert a != r

    def test_access_decodable_as_access(self):
        a, _, _ = issue_token_pair(secret=SECRET, sub="x@y", uid=1,
                                     role="viewer")
        decode_token(a, SECRET, expected_type="access")   # no raise

    def test_refresh_decodable_as_refresh(self):
        _, r, _ = issue_token_pair(secret=SECRET, sub="x@y", uid=1,
                                     role="viewer")
        decode_token(r, SECRET, expected_type="refresh")

    def test_refresh_rejected_as_access(self):
        _, r, _ = issue_token_pair(secret=SECRET, sub="x@y", uid=1,
                                     role="viewer")
        with pytest.raises(TokenTypeMismatchError):
            decode_token(r, SECRET, expected_type="access")

    def test_expires_in_matches_access_ttl(self):
        _, _, exp_in = issue_token_pair(secret=SECRET, sub="x", uid=1,
                                          role="viewer", access_ttl=300)
        assert exp_in == 300


class TestExtractBearer:
    def test_extracts_token(self):
        assert extract_bearer_token("Bearer abc.def.ghi") == "abc.def.ghi"

    def test_case_insensitive_scheme(self):
        assert extract_bearer_token("bearer xyz") == "xyz"

    def test_returns_none_for_missing(self):
        assert extract_bearer_token(None) is None
        assert extract_bearer_token("")   is None

    def test_returns_none_for_wrong_scheme(self):
        assert extract_bearer_token("Basic xxx") is None

    def test_returns_none_for_malformed(self):
        assert extract_bearer_token("Bearer") is None
        assert extract_bearer_token("Bearer too many parts") is None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_auth_env(monkeypatch):
    for v in ("AXION_API_KEY", "AXION_API_KEY_VIEWER", "AXION_API_KEY_OPERATOR",
              "AXION_API_KEY_MANAGER", "AXION_JWT_SECRET",
              "AXION_JWT_ACCESS_MINUTES", "AXION_JWT_REFRESH_DAYS"):
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app, raise_server_exceptions=False)


def _stub_user(uid=1, email="alice@axion.test", role="manager",
                password="correct-horse-battery-staple", active=True):
    """Build a UserRecord-like MagicMock that the repo will return."""
    user = MagicMock()
    user.id            = uid
    user.email         = email
    user.role          = role
    user.active        = active
    user.password_hash = hash_password(password)
    user.to_public_dict.return_value = {
        "id": uid, "email": email, "role": role, "active": active,
        "created_at": None,
    }
    return user


@pytest.fixture
def with_user(monkeypatch):
    """Patch UserRepository.get_by_email/id so the endpoints don't hit DB."""
    user = _stub_user()
    repo = MagicMock()
    repo.get_by_email.side_effect = lambda email: (
        user if email == user.email else None
    )
    repo.get_by_id.side_effect = lambda uid: (
        user if uid == user.id else None
    )
    # Patch the class so any instantiation returns our stub
    from api import server as srv
    monkeypatch.setattr(srv, "UserRepository", lambda db: repo)
    monkeypatch.setattr(srv.state, "db", MagicMock())   # any non-None object
    return user


class TestLoginEndpoint:
    def test_503_when_jwt_secret_unset(self, client, with_user):
        r = client.post("/api/auth/login",
                         json={"email": "alice@axion.test", "password": "x"})
        assert r.status_code == 503

    def test_503_when_db_unavailable(self, client, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        from api import server as srv
        monkeypatch.setattr(srv.state, "db", None)
        r = client.post("/api/auth/login",
                         json={"email": "x@y.com", "password": "z"})
        assert r.status_code == 503

    def test_400_when_email_or_password_missing(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        r = client.post("/api/auth/login", json={"email": "x"})
        assert r.status_code == 400

    def test_401_when_user_not_found(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        r = client.post("/api/auth/login",
                         json={"email": "nope@axion.test", "password": "x"})
        assert r.status_code == 401

    def test_401_when_password_wrong(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        r = client.post("/api/auth/login",
                         json={"email": "alice@axion.test", "password": "wrong"})
        assert r.status_code == 401

    def test_200_returns_tokens_on_success(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        r = client.post("/api/auth/login", json={
            "email": "alice@axion.test",
            "password": "correct-horse-battery-staple",
        })
        assert r.status_code == 200
        body = r.json()
        for k in ("access_token", "refresh_token", "expires_in", "user"):
            assert k in body
        assert body["user"]["role"] == "manager"

    def test_inactive_user_rejected(self, client, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        user = _stub_user(active=False)
        repo = MagicMock(); repo.get_by_email.return_value = user
        from api import server as srv
        monkeypatch.setattr(srv, "UserRepository", lambda db: repo)
        monkeypatch.setattr(srv.state, "db", MagicMock())
        r = client.post("/api/auth/login", json={
            "email": user.email, "password": "correct-horse-battery-staple",
        })
        assert r.status_code == 401


class TestRefreshEndpoint:
    def test_400_when_token_missing(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        r = client.post("/api/auth/refresh", json={})
        assert r.status_code == 400

    def test_401_on_invalid_token(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        r = client.post("/api/auth/refresh", json={"refresh_token": "garbage"})
        assert r.status_code == 401

    def test_401_when_access_token_passed(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        access, _, _ = issue_token_pair(secret=SECRET, sub="x", uid=1,
                                          role="manager")
        r = client.post("/api/auth/refresh", json={"refresh_token": access})
        assert r.status_code == 401

    def test_200_issues_new_pair(self, client, with_user, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        _, refresh, _ = issue_token_pair(
            secret=SECRET, sub=with_user.email, uid=with_user.id,
            role=with_user.role,
        )
        r = client.post("/api/auth/refresh",
                         json={"refresh_token": refresh})
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body and "refresh_token" in body


class TestMeEndpoint:
    def test_401_without_token_when_jwt_enabled(self, client, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_returns_user_info_with_valid_token(self, client, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        access, _, _ = issue_token_pair(secret=SECRET, sub="alice@axion.test",
                                          uid=42, role="operator")
        r = client.get("/api/auth/me",
                        headers={"Authorization": f"Bearer {access}"})
        assert r.status_code == 200
        body = r.json()
        assert body["uid"]   == 42
        assert body["role"]  == "operator"
        assert body["email"] == "alice@axion.test"


# ─────────────────────────────────────────────────────────────────────────────
# Middleware: JWT integrates with role hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestMiddlewareWithJwt:
    def test_jwt_viewer_can_get(self, client, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        access, _, _ = issue_token_pair(secret=SECRET, sub="v@x", uid=1,
                                          role="viewer")
        r = client.get("/api/scenarios",
                        headers={"Authorization": f"Bearer {access}"})
        assert r.status_code != 401
        assert r.status_code != 403

    def test_jwt_viewer_cannot_post_decision(self, client, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        access, _, _ = issue_token_pair(secret=SECRET, sub="v@x", uid=1,
                                          role="viewer")
        r = client.post(
            "/api/recommendations/REC-1/decide",
            headers={"Authorization": f"Bearer {access}"},
            json={"action": "accept"},
        )
        assert r.status_code == 403

    def test_expired_token_returns_401_token_expired(self, client, monkeypatch):
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        past = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())
        token = encode_token(secret=SECRET, sub="x", uid=1, role="manager",
                              token_type="access", ttl_seconds=60,
                              issued_at=past - 60)
        r = client.get("/api/scenarios",
                        headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401
        body = r.json()
        assert body.get("code") == "token_expired"

    def test_jwt_takes_precedence_over_api_key(self, client, monkeypatch):
        # Both backends configured: JWT viewer should override API-key manager
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        monkeypatch.setenv("AXION_API_KEY_MANAGER", "m-key")
        access, _, _ = issue_token_pair(secret=SECRET, sub="v@x", uid=1,
                                          role="viewer")
        r = client.post(
            "/api/scenarios/select",
            headers={"Authorization": f"Bearer {access}",
                     "X-API-Key":     "m-key"},
            json={"scenario": "thermal_drift"},
        )
        # viewer cannot change scenario → 403, even though api-key alone
        # would have been allowed
        assert r.status_code == 403

    def test_login_path_is_public(self, client, monkeypatch):
        # JWT enabled but no Authorization header — login must still be hit
        monkeypatch.setenv("AXION_JWT_SECRET", SECRET)
        # No DB stub → endpoint will 503, but it should be reachable (not 401)
        from api import server as srv
        monkeypatch.setattr(srv.state, "db", None)
        r = client.post("/api/auth/login",
                         json={"email": "x@y.com", "password": "z"})
        assert r.status_code != 401
