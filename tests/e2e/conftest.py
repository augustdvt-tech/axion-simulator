"""
Shared fixtures for end-to-end tests.

The HTTP-only suite spawns a real uvicorn process pointing at the live
FastAPI app, then drives it with `httpx`. This catches real wiring bugs
that unit-level TestClient misses (startup hooks, replay loop scheduling,
WebSocket lifecycle, middleware ordering across processes).

The browser suite (test_dashboard_e2e.py) is opt-in and gated behind
`--run-browser` so CI doesn't pull Playwright by default — see usage in
that file's module docstring.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def pytest_addoption(parser):
    parser.addoption(
        "--run-browser", action="store_true", default=False,
        help="Run the Playwright browser tests (requires playwright install).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests that talk to a live uvicorn server",
    )
    config.addinivalue_line(
        "markers", "browser: browser-driven E2E test (gated by --run-browser)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-browser"):
        return
    skip_browser = pytest.mark.skip(reason="needs --run-browser")
    for item in items:
        if "browser" in item.keywords:
            item.add_marker(skip_browser)


# ─────────────────────────────────────────────────────────────────────────────
# Free port discovery
# ─────────────────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ─────────────────────────────────────────────────────────────────────────────
# Live server fixture (session-scoped — one process for the whole module)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def live_server() -> Iterator[str]:
    """Start a uvicorn subprocess pointing at api.server:app, yield the base URL,
    then terminate. Auth is disabled (no AXION_API_KEY / AXION_JWT_SECRET) so
    tests don't need to deal with credentials."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    # Hermetic: no auth, no rate limit, no DB, no OPC-UA, no JWT, no webhook
    for v in (
        "AXION_API_KEY", "AXION_API_KEY_VIEWER", "AXION_API_KEY_OPERATOR",
        "AXION_API_KEY_MANAGER", "AXION_JWT_SECRET", "AXION_DB_URL",
        "AXION_OPCUA_ENABLED", "AXION_WEBHOOK_URL",
    ):
        env.pop(v, None)
    env["AXION_RATE_LIMIT_PER_MIN"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Poll /api/health until ready (max ~15s)
    deadline = time.time() + 15.0
    last_err = None
    while time.time() < deadline:
        if proc.poll() is not None:
            out = (proc.stdout.read() or b"").decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"uvicorn died during startup:\n{out}")
        try:
            r = httpx.get(f"{base_url}/api/health", timeout=1.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError as e:
            last_err = e
        time.sleep(0.2)
    else:
        proc.terminate()
        out = (proc.stdout.read() or b"").decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"server did not become ready: {last_err}\n{out}")

    try:
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def http_client(live_server: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=live_server, timeout=30.0) as c:
        yield c
