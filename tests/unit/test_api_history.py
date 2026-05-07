"""Tests for /api/history/* endpoints."""

import sys
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, ".")


@pytest.fixture
def client():
    from api.server import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_db(monkeypatch):
    """Inject a mock DbClient into server state."""
    from api import server
    db = MagicMock()
    db.list_ingested_scenarios.return_value = [
        {"name": "thermal_drift", "n_samples": 1440, "ingested_at": "2026-01-01T00:00:00"}
    ]
    db.query_samples.return_value = [
        {"timestamp": "2026-01-01T00:00:00", "scenario": "thermal_drift",
         "cstr.T_R_C": 79.1, "column.purity_B": 98.8}
    ]
    db.query_recommendations.return_value = [
        {"id": "REC-001", "scenario": "thermal_drift", "urgency": "high",
         "rule_id": "R01_ThermalDrift", "status": "pending",
         "timestamp": "2026-01-01T00:30:00"}
    ]
    db.query_decisions.return_value = [
        {"id": 1, "recommendation_id": "REC-001", "decision": "accepted",
         "rationale": "OK", "decided_at": "2026-01-01T00:35:00",
         "scenario": "thermal_drift"}
    ]
    monkeypatch.setattr(server.state, "db", db)
    return db


class TestHistoryWithoutDb:
    def test_scenarios_503_without_db(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "db", None)
        r = client.get("/api/history/scenarios")
        assert r.status_code == 503

    def test_samples_503_without_db(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "db", None)
        r = client.get("/api/history/samples")
        assert r.status_code == 503

    def test_recommendations_503_without_db(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "db", None)
        r = client.get("/api/history/recommendations")
        assert r.status_code == 503

    def test_decisions_503_without_db(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "db", None)
        r = client.get("/api/history/decisions")
        assert r.status_code == 503


class TestHistoryScenarios:
    def test_returns_list(self, client, mock_db):
        r = client.get("/api/history/scenarios")
        assert r.status_code == 200
        assert "scenarios" in r.json()
        assert isinstance(r.json()["scenarios"], list)

    def test_scenario_has_name(self, client, mock_db):
        r = client.get("/api/history/scenarios")
        assert r.json()["scenarios"][0]["name"] == "thermal_drift"


class TestHistorySamples:
    def test_returns_count_and_data(self, client, mock_db):
        r = client.get("/api/history/samples")
        assert r.status_code == 200
        body = r.json()
        assert "count" in body
        assert "data" in body
        assert body["count"] == len(body["data"])

    def test_scenario_filter_passed_to_db(self, client, mock_db):
        client.get("/api/history/samples?scenario=thermal_drift&limit=50")
        mock_db.query_samples.assert_called_once()
        kwargs = mock_db.query_samples.call_args.kwargs
        assert kwargs["scenario"] == "thermal_drift"
        assert kwargs["limit"] == 50

    def test_tags_filter_parsed(self, client, mock_db):
        client.get("/api/history/samples?tags=cstr.T_R_C,column.purity_B")
        kwargs = mock_db.query_samples.call_args.kwargs
        assert kwargs["tags"] == ["cstr.T_R_C", "column.purity_B"]

    def test_timestamp_filters_passed(self, client, mock_db):
        client.get("/api/history/samples?from_ts=2026-01-01T00:00:00&to_ts=2026-01-01T01:00:00")
        kwargs = mock_db.query_samples.call_args.kwargs
        assert kwargs["from_ts"] == "2026-01-01T00:00:00"
        assert kwargs["to_ts"] == "2026-01-01T01:00:00"


class TestHistoryRecommendations:
    def test_returns_count_and_data(self, client, mock_db):
        r = client.get("/api/history/recommendations")
        assert r.status_code == 200
        body = r.json()
        assert "count" in body and "data" in body

    def test_status_filter_parsed(self, client, mock_db):
        client.get("/api/history/recommendations?status=pending,accepted")
        kwargs = mock_db.query_recommendations.call_args.kwargs
        assert "pending" in kwargs["status"]
        assert "accepted" in kwargs["status"]

    def test_urgency_filter_parsed(self, client, mock_db):
        client.get("/api/history/recommendations?urgency=high,critical")
        kwargs = mock_db.query_recommendations.call_args.kwargs
        assert "high" in kwargs["urgency"]
        assert "critical" in kwargs["urgency"]


class TestHistoryDecisions:
    def test_returns_count_and_data(self, client, mock_db):
        r = client.get("/api/history/decisions")
        assert r.status_code == 200
        body = r.json()
        assert "count" in body and "data" in body

    def test_scenario_filter_passed(self, client, mock_db):
        client.get("/api/history/decisions?scenario=feed_perturbation")
        kwargs = mock_db.query_decisions.call_args.kwargs
        assert kwargs["scenario"] == "feed_perturbation"
