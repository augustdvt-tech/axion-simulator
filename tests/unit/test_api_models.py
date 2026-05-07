"""Tests for GET /api/models/status endpoint."""

import sys
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, ".")


@pytest.fixture
def client():
    from api.server import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def patch_models_dir(tmp_path, monkeypatch):
    """Point MODELS_DIR at a temp directory so tests don't touch real files."""
    import api.server as server
    monkeypatch.setattr(server, "MODELS_DIR", tmp_path)
    return tmp_path


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh)


# ─────────────────────────────────────────────────────────────────────────────
# Status when nothing is trained
# ─────────────────────────────────────────────────────────────────────────────

class TestModelsStatusNotTrained:
    def test_returns_200(self, client, patch_models_dir):
        r = client.get("/api/models/status")
        assert r.status_code == 200

    def test_soft_sensor_not_trained(self, client, patch_models_dir):
        body = client.get("/api/models/status").json()
        assert body["soft_sensor"]["status"] == "not_trained"

    def test_lstm_not_trained(self, client, patch_models_dir):
        body = client.get("/api/models/status").json()
        assert body["lstm_forecaster"]["status"] == "not_trained"

    def test_soft_sensor_metrics_is_none(self, client, patch_models_dir):
        body = client.get("/api/models/status").json()
        assert body["soft_sensor"]["metrics"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Status when soft sensor is trained (with metrics JSON)
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftSensorTrained:
    METRICS = {
        "mae_holdout":  0.082,
        "mae_overall":  0.071,
        "rmse_overall": 0.098,
        "r2_train":     0.97,
        "n_samples":    10080,
        "n_ensemble":   5,
    }

    @pytest.fixture(autouse=True)
    def _write(self, patch_models_dir):
        _write_metrics(patch_models_dir / "purity_soft_sensor.metrics.json",
                       self.METRICS)

    def test_status_is_trained(self, client):
        body = client.get("/api/models/status").json()
        assert body["soft_sensor"]["status"] == "trained"

    def test_metrics_returned(self, client):
        body = client.get("/api/models/status").json()
        m = body["soft_sensor"]["metrics"]
        assert m["mae_holdout"] == pytest.approx(0.082)
        assert m["n_samples"] == 10080

    def test_trained_at_is_iso_string(self, client):
        body = client.get("/api/models/status").json()
        trained_at = body["soft_sensor"]["trained_at"]
        assert "T" in trained_at   # ISO 8601

    def test_all_metric_keys_present(self, client):
        body = client.get("/api/models/status").json()
        m = body["soft_sensor"]["metrics"]
        for key in ("mae_holdout", "mae_overall", "r2_train", "n_samples"):
            assert key in m, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Status when only the model file exists (no metrics JSON)
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftSensorTrainedNoMetrics:
    @pytest.fixture(autouse=True)
    def _write(self, patch_models_dir):
        p = patch_models_dir / "purity_soft_sensor.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fake-model")

    def test_status_is_trained(self, client):
        body = client.get("/api/models/status").json()
        assert body["soft_sensor"]["status"] == "trained"

    def test_metrics_is_none(self, client):
        body = client.get("/api/models/status").json()
        assert body["soft_sensor"]["metrics"] is None

    def test_trained_at_present(self, client):
        body = client.get("/api/models/status").json()
        assert body["soft_sensor"].get("trained_at") is not None


# ─────────────────────────────────────────────────────────────────────────────
# LSTM forecaster status
# ─────────────────────────────────────────────────────────────────────────────

class TestLSTMStatus:
    @pytest.fixture(autouse=True)
    def _create_dir(self, patch_models_dir):
        lstm_dir = patch_models_dir / "lstm_forecaster"
        lstm_dir.mkdir(parents=True, exist_ok=True)
        (lstm_dir / "model.keras").write_text("fake-lstm")

    def test_lstm_status_trained(self, client):
        body = client.get("/api/models/status").json()
        assert body["lstm_forecaster"]["status"] == "trained"

    def test_lstm_trained_at_present(self, client):
        body = client.get("/api/models/status").json()
        assert "trained_at" in body["lstm_forecaster"]


# ─────────────────────────────────────────────────────────────────────────────
# Both models trained
# ─────────────────────────────────────────────────────────────────────────────

class TestBothTrained:
    @pytest.fixture(autouse=True)
    def _setup(self, patch_models_dir):
        _write_metrics(patch_models_dir / "purity_soft_sensor.metrics.json",
                       {"mae_holdout": 0.1})
        lstm_dir = patch_models_dir / "lstm_forecaster"
        lstm_dir.mkdir(parents=True, exist_ok=True)

    def test_response_has_both_keys(self, client):
        body = client.get("/api/models/status").json()
        assert "soft_sensor" in body
        assert "lstm_forecaster" in body

    def test_both_trained(self, client):
        body = client.get("/api/models/status").json()
        assert body["soft_sensor"]["status"] == "trained"
        assert body["lstm_forecaster"]["status"] == "trained"
