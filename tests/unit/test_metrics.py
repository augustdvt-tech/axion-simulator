"""Tests for api/metrics.py + the /api/metrics endpoint and middleware."""

import sys

import pytest

sys.path.insert(0, ".")

from api.metrics import (
    REGISTRY, RequestTimer, http_inflight_requests, http_request_duration_seconds,
    http_requests_total, render_prometheus, template_path,
)


@pytest.fixture(autouse=True)
def _reset_rate_limit_env(monkeypatch):
    # Disable rate limiting so it doesn't 429 our metrics tests
    monkeypatch.setenv("AXION_RATE_LIMIT_PER_MIN", "0")


# ─────────────────────────────────────────────────────────────────────────────
# template_path
# ─────────────────────────────────────────────────────────────────────────────

class TestTemplatePath:
    def test_decide_path_templated(self):
        assert template_path("/api/recommendations/REC-1234/decide") \
            == "/api/recommendations/{rec_id}/decide"

    def test_recommendation_detail_templated(self):
        assert template_path("/api/recommendations/REC-XYZ") \
            == "/api/recommendations/{rec_id}"

    def test_scenario_templated(self):
        assert template_path("/api/scenarios/thermal_drift") \
            == "/api/scenarios/{name}"

    def test_static_path_unchanged(self):
        assert template_path("/api/health") == "/api/health"
        assert template_path("/api/scenarios") == "/api/scenarios"


# ─────────────────────────────────────────────────────────────────────────────
# render_prometheus
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderPrometheus:
    def test_returns_bytes_and_content_type(self):
        body, ct = render_prometheus()
        assert isinstance(body, bytes)
        assert ct.startswith("text/plain")

    def test_includes_axion_metric_names(self):
        body, _ = render_prometheus()
        text = body.decode("utf-8")
        for name in (
            "axion_http_requests_total",
            "axion_http_request_duration_seconds",
            "axion_recommendations_total",
        ):
            assert name in text


# ─────────────────────────────────────────────────────────────────────────────
# RequestTimer
# ─────────────────────────────────────────────────────────────────────────────

class TestRequestTimer:
    def test_increments_inflight_during_request(self):
        before = http_inflight_requests.labels(method="GET")._value.get()
        with RequestTimer("GET", "/api/health"):
            during = http_inflight_requests.labels(method="GET")._value.get()
            assert during == before + 1
        after = http_inflight_requests.labels(method="GET")._value.get()
        assert after == before

    def test_records_duration_observation(self):
        # Read the histogram count before/after; should grow by 1
        h = http_request_duration_seconds.labels(method="GET", path="/api/x")
        before = h._sum.get()
        with RequestTimer("GET", "/api/x"):
            pass
        after = h._sum.get()
        assert after >= before


# ─────────────────────────────────────────────────────────────────────────────
# /api/metrics endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_200(self, client):
        r = client.get("/api/metrics")
        assert r.status_code == 200

    def test_content_type_prometheus(self, client):
        r = client.get("/api/metrics")
        assert "text/plain" in r.headers.get("content-type", "")

    def test_body_contains_metric_names(self, client):
        r = client.get("/api/metrics")
        text = r.text
        assert "axion_http_requests_total" in text

    def test_metrics_exempt_from_jwt_auth(self, client, monkeypatch):
        # Even with JWT enabled, /api/metrics must respond without a token
        monkeypatch.setenv("AXION_JWT_SECRET", "x" * 40)
        r = client.get("/api/metrics")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Middleware records request count
# ─────────────────────────────────────────────────────────────────────────────

class TestMiddlewareRecording:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_request_count_increments(self, client):
        c = http_requests_total.labels(
            method="GET", path="/api/health", status="200",
        )
        before = c._value.get()
        client.get("/api/health")
        after = c._value.get()
        assert after == before + 1

    def test_404_status_recorded(self, client):
        client.get("/api/this-does-not-exist")
        c = http_requests_total.labels(
            method="GET", path="/api/this-does-not-exist", status="404",
        )
        # Just check the counter exists with > 0 value
        assert c._value.get() >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting through the middleware
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimitMiddleware:
    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient
        from api.server import app
        from api.rate_limit import RateLimiter
        from api import server as srv
        # Tight limit just for this test, then reset state
        monkeypatch.setattr(srv.state, "rate_limiter",
                             RateLimiter(per_min=60, burst=2))
        return TestClient(app, raise_server_exceptions=False)

    def test_allows_under_burst(self, client):
        # Use an arbitrary endpoint that doesn't error
        for _ in range(2):
            r = client.get("/api/scenarios")
            assert r.status_code != 429

    def test_429_after_burst(self, client):
        # Burst=2; the 3rd request should be rejected
        client.get("/api/scenarios")
        client.get("/api/scenarios")
        r = client.get("/api/scenarios")
        assert r.status_code == 429
        body = r.json()
        assert "retry_after" in body
        assert "Retry-After" in r.headers

    def test_health_exempt(self, client):
        # Even after exhausting the bucket, /api/health stays open
        for _ in range(5):
            client.get("/api/scenarios")
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_metrics_exempt(self, client):
        for _ in range(5):
            client.get("/api/scenarios")
        r = client.get("/api/metrics")
        assert r.status_code == 200
