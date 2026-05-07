"""
Axion AI — Prometheus metrics
=============================

Exposes a single instance of the Prometheus default registry plus a tidy
set of metrics that the rest of the platform records into.

What we measure
---------------
http_requests_total          counter, labels: method, path_template, status
http_request_duration_seconds histogram, labels: method, path_template
http_inflight_requests       gauge, labels: method
recommendations_total        counter, labels: rule, urgency, scenario
operator_decisions_total     counter, labels: status (accepted | rejected | modified)
websocket_connections        gauge
rate_limit_rejections_total  counter, labels: limiter (ip | user)

Exposition
----------
The `/api/metrics` endpoint returns the standard Prometheus text format,
ready to be scraped by a Prometheus server. It is exempt from RBAC so a
scraper without an API key can poll it (the metrics themselves are not
sensitive — no PII, no secrets).

Path templating
---------------
Raw URLs like `/api/recommendations/REC-1234/decide` would explode the
cardinality of the `path` label. We map dynamic path segments to a
template form (`/api/recommendations/{rec_id}/decide`) so each endpoint
collapses to a single time series. Mappings live in `_TEMPLATES` below
and are applied in `template_path()`.
"""

from __future__ import annotations

import re
import time
from typing import Tuple

from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram,
    generate_latest,
)


# Use the default registry so prometheus_client's text format includes
# any metrics third-party libs might register too. For tests that need
# isolation, callers can pass `registry=` to `_make_*` factories.
REGISTRY = CollectorRegistry()


def _make_counter(name: str, doc: str, labels=()):
    return Counter(name, doc, labels, registry=REGISTRY)


def _make_histogram(name: str, doc: str, labels=(), buckets=None):
    kwargs = {"registry": REGISTRY}
    if buckets is not None:
        kwargs["buckets"] = buckets
    return Histogram(name, doc, labels, **kwargs)


def _make_gauge(name: str, doc: str, labels=()):
    return Gauge(name, doc, labels, registry=REGISTRY)


# Latency buckets in seconds — geometric-ish, weighted toward sub-second.
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric instances
# ─────────────────────────────────────────────────────────────────────────────

http_requests_total = _make_counter(
    "axion_http_requests_total",
    "Total HTTP requests served, labeled by method, path template and status.",
    ["method", "path", "status"],
)

http_request_duration_seconds = _make_histogram(
    "axion_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
    buckets=_LATENCY_BUCKETS,
)

http_inflight_requests = _make_gauge(
    "axion_http_inflight_requests",
    "HTTP requests currently being processed.",
    ["method"],
)

recommendations_total = _make_counter(
    "axion_recommendations_total",
    "Total recommendations issued, labeled by rule, urgency and scenario.",
    ["rule", "urgency", "scenario"],
)

operator_decisions_total = _make_counter(
    "axion_operator_decisions_total",
    "Total operator decisions recorded, labeled by status.",
    ["status"],
)

websocket_connections = _make_gauge(
    "axion_websocket_connections",
    "Number of currently-connected WebSocket clients.",
)

rate_limit_rejections_total = _make_counter(
    "axion_rate_limit_rejections_total",
    "Requests rejected by the rate limiter, labeled by limiter type.",
    ["limiter"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Path templating
# ─────────────────────────────────────────────────────────────────────────────

# Each entry is (regex, replacement). Tested against the request path
# in declaration order; first match wins.
_TEMPLATES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"^/api/recommendations/[^/]+/decide$"),
                "/api/recommendations/{rec_id}/decide"),
    (re.compile(r"^/api/recommendations/[^/]+$"),
                "/api/recommendations/{rec_id}"),
    (re.compile(r"^/api/scenarios/[^/]+$"),
                "/api/scenarios/{name}"),
)


def template_path(path: str) -> str:
    for pattern, repl in _TEMPLATES:
        if pattern.match(path):
            return repl
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Exposition
# ─────────────────────────────────────────────────────────────────────────────

def render_prometheus() -> Tuple[bytes, str]:
    """Render the registry as Prometheus text exposition. Returns (body, content_type)."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# ─────────────────────────────────────────────────────────────────────────────
# Convenience timer
# ─────────────────────────────────────────────────────────────────────────────

class RequestTimer:
    """Context manager that records inflight + duration metrics for one request."""

    def __init__(self, method: str, path_template: str):
        self.method = method
        self.path = path_template
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        http_inflight_requests.labels(method=self.method).inc()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._start
        http_request_duration_seconds.labels(
            method=self.method, path=self.path,
        ).observe(elapsed)
        http_inflight_requests.labels(method=self.method).dec()
        return False
