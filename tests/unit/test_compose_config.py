"""Sanity tests for docker-compose.yml + Dockerfile (Bloque Y).

These don't actually build images or boot containers — they just verify
that the compose manifest is well-formed and references the services we
expect. Catches typos in env var names, missing depends_on, and broken
volume mounts before the developer wastes a `docker compose up`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def compose():
    yaml = pytest.importorskip("yaml")
    raw = (PROJECT_ROOT / "docker-compose.yml").read_text()
    return yaml.safe_load(raw)


@pytest.fixture(scope="module")
def dockerfile():
    return (PROJECT_ROOT / "Dockerfile").read_text()


# ─────────────────────────────────────────────────────────────────────────────
# Compose structure
# ─────────────────────────────────────────────────────────────────────────────

class TestComposeStructure:
    def test_has_required_services(self, compose):
        services = compose.get("services", {})
        for name in ("timescaledb", "mlflow", "axion-api", "migrate"):
            assert name in services, f"Missing service: {name}"

    def test_axion_api_builds_from_dockerfile(self, compose):
        api = compose["services"]["axion-api"]
        assert "build" in api
        assert api["build"]["dockerfile"] == "Dockerfile"

    def test_axion_api_depends_on_db_and_migrate(self, compose):
        api = compose["services"]["axion-api"]
        deps = api.get("depends_on", {})
        # Both styles supported by compose; we use object form
        assert "timescaledb" in deps
        assert "migrate"     in deps
        assert "mlflow"      in deps

    def test_migrate_runs_alembic(self, compose):
        m = compose["services"]["migrate"]
        cmd = m.get("command", [])
        assert "alembic" in " ".join(map(str, cmd))
        assert "upgrade" in " ".join(map(str, cmd))

    def test_db_url_points_at_compose_service(self, compose):
        api = compose["services"]["axion-api"]
        env = api.get("environment", {})
        url = env.get("AXION_DB_URL", "")
        assert "@timescaledb:" in url, \
            "axion-api should connect to the timescaledb service by name"

    def test_mlflow_tracking_uri_points_at_service(self, compose):
        api = compose["services"]["axion-api"]
        env = api.get("environment", {})
        assert env.get("MLFLOW_TRACKING_URI", "").endswith(":5000")
        assert "mlflow" in env["MLFLOW_TRACKING_URI"]

    def test_axion_api_port_8000_published(self, compose):
        api = compose["services"]["axion-api"]
        assert any("8000" in p for p in api.get("ports", []))

    def test_volumes_declared(self, compose):
        vols = compose.get("volumes", {})
        assert "pgdata" in vols
        assert "mlruns" in vols

    def test_axion_api_has_healthcheck(self, compose):
        api = compose["services"]["axion-api"]
        hc = api.get("healthcheck", {})
        assert hc, "axion-api should have a healthcheck"
        cmd = " ".join(map(str, hc.get("test", [])))
        assert "/api/health" in cmd


# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile shape
# ─────────────────────────────────────────────────────────────────────────────

class TestDockerfile:
    def test_multistage_build(self, dockerfile):
        # Builder + runtime stages
        stages = [l for l in dockerfile.splitlines() if l.startswith("FROM ")]
        assert len(stages) >= 2

    def test_runs_as_non_root(self, dockerfile):
        assert "USER axion" in dockerfile

    def test_exposes_8000(self, dockerfile):
        assert "EXPOSE 8000" in dockerfile

    def test_has_healthcheck(self, dockerfile):
        assert "HEALTHCHECK" in dockerfile
        assert "/api/health" in dockerfile

    def test_uvicorn_entrypoint(self, dockerfile):
        # CMD line invokes uvicorn against api.server:app
        assert "uvicorn" in dockerfile
        assert "api.server:app" in dockerfile


# ─────────────────────────────────────────────────────────────────────────────
# .dockerignore — keep the image lean
# ─────────────────────────────────────────────────────────────────────────────

class TestDockerignore:
    @pytest.fixture(scope="class")
    def ignore(self):
        path = PROJECT_ROOT / ".dockerignore"
        if not path.exists():
            pytest.fail(".dockerignore missing")
        return path.read_text().splitlines()

    def test_excludes_git_history(self, ignore):
        assert ".git" in ignore

    def test_excludes_pycache(self, ignore):
        assert "__pycache__" in ignore

    def test_excludes_venv(self, ignore):
        assert any(v in ignore for v in (".venv", "venv"))

    def test_excludes_env_files(self, ignore):
        assert ".env" in ignore or "*.env" in ignore

    def test_excludes_tests(self, ignore):
        # Image doesn't need the test suite at runtime
        assert "tests" in ignore
