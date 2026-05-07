"""Tests for analytics/drift.py — PSI helpers, DriftDetector, /api/drift/status."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from analytics.drift import (
    DriftDetector,
    DriftReport,
    classify_psi,
    compute_psi,
    quantile_bin_edges,
    PSI_THRESHOLD_MODERATE,
    PSI_THRESHOLD_SIGNIFICANT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ref_df(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "column.T_bot_C":   rng.normal(120.0, 2.0, n),
        "column.RR":        rng.normal(5.5,   0.2, n),
        "cstr.C_A":         rng.normal(2.0,   0.1, n),
    })


def _shifted_df(n=500, t_shift=8.0, seed=1) -> pd.DataFrame:
    """Same shape as _ref_df but column.T_bot_C is shifted by t_shift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "column.T_bot_C":   rng.normal(120.0 + t_shift, 2.0, n),
        "column.RR":        rng.normal(5.5,             0.2, n),
        "cstr.C_A":         rng.normal(2.0,             0.1, n),
    })


# ─────────────────────────────────────────────────────────────────────────────
# classify_psi
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyPsi:
    def test_zero_is_none(self):
        assert classify_psi(0.0) == "none"

    def test_just_below_moderate_is_none(self):
        assert classify_psi(PSI_THRESHOLD_MODERATE - 0.01) == "none"

    def test_at_moderate_is_moderate(self):
        assert classify_psi(PSI_THRESHOLD_MODERATE) == "moderate"

    def test_just_below_significant_is_moderate(self):
        assert classify_psi(PSI_THRESHOLD_SIGNIFICANT - 0.01) == "moderate"

    def test_at_significant_is_significant(self):
        assert classify_psi(PSI_THRESHOLD_SIGNIFICANT) == "significant"

    def test_large_is_significant(self):
        assert classify_psi(1.0) == "significant"


# ─────────────────────────────────────────────────────────────────────────────
# quantile_bin_edges
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantileBinEdges:
    def test_n_bins_returns_n_plus_one_edges(self):
        edges = quantile_bin_edges(np.linspace(0, 1, 1000), n_bins=10)
        assert len(edges) == 11

    def test_strictly_increasing(self):
        edges = quantile_bin_edges(np.random.default_rng(0).normal(0, 1, 1000),
                                    n_bins=10)
        diffs = np.diff(edges)
        assert (diffs > 0).all()

    def test_constant_input_returns_two_edges(self):
        edges = quantile_bin_edges(np.array([5.0]))
        assert len(edges) == 2
        assert edges[0] < edges[1]

    def test_handles_nan(self):
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        edges = quantile_bin_edges(x, n_bins=4)
        assert len(edges) == 5

    def test_constant_column_strictly_increasing(self):
        edges = quantile_bin_edges(np.full(100, 7.0), n_bins=5)
        assert (np.diff(edges) > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# compute_psi
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePsi:
    def test_identical_distributions_psi_near_zero(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        # Sample again from the same distribution
        live = rng.normal(0, 1, 5000)
        edges = quantile_bin_edges(ref, n_bins=10)
        psi = compute_psi(ref, live, edges)
        assert psi < 0.05   # statistical noise only

    def test_shifted_mean_increases_psi(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        live = rng.normal(2.0, 1, 5000)   # shift by 2 sigma
        edges = quantile_bin_edges(ref, n_bins=10)
        psi = compute_psi(ref, live, edges)
        assert psi > 0.5

    def test_extreme_shift_gives_significant_psi(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        live = rng.normal(10, 1, 5000)
        edges = quantile_bin_edges(ref, n_bins=10)
        psi = compute_psi(ref, live, edges)
        assert psi >= PSI_THRESHOLD_SIGNIFICANT

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            ref = rng.normal(0, 1, 500)
            live = rng.normal(rng.uniform(-2, 2), rng.uniform(0.5, 2), 500)
            edges = quantile_bin_edges(ref, n_bins=10)
            assert compute_psi(ref, live, edges) >= 0.0

    def test_empty_inputs_return_zero(self):
        edges = np.array([0.0, 1.0, 2.0])
        assert compute_psi(np.array([]), np.array([1.0]), edges) == 0.0
        assert compute_psi(np.array([1.0]), np.array([]), edges) == 0.0

    def test_handles_nans(self):
        ref = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0])
        live = np.array([1.0, 2.0, np.nan, 3.0])
        edges = quantile_bin_edges(ref, n_bins=3)
        psi = compute_psi(ref, live, edges)
        assert np.isfinite(psi)
        assert psi >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DriftDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetector:
    def test_fit_marks_detector_fitted(self):
        d = DriftDetector(features=["column.T_bot_C"])
        assert d.fitted is False
        d.fit(_ref_df())
        assert d.fitted is True

    def test_score_before_fit_raises(self):
        d = DriftDetector(features=["column.T_bot_C"])
        with pytest.raises(RuntimeError):
            d.score(_ref_df())

    def test_no_drift_on_resampled_reference(self):
        d = DriftDetector(features=["column.T_bot_C", "column.RR"]).fit(_ref_df())
        report = d.score(_ref_df(seed=1))
        assert report.overall_status == "none"

    def test_significant_drift_detected_on_shifted_data(self):
        d = DriftDetector(features=["column.T_bot_C", "column.RR"]).fit(_ref_df())
        report = d.score(_shifted_df(t_shift=10.0))
        assert report.overall_status == "significant"
        assert report.worst_feature == "column.T_bot_C"

    def test_unaffected_feature_stays_none(self):
        d = DriftDetector(features=["column.T_bot_C", "column.RR"]).fit(_ref_df())
        report = d.score(_shifted_df(t_shift=10.0))
        rr = next(f for f in report.by_feature if f.feature == "column.RR")
        assert rr.status == "none"

    def test_missing_feature_in_live_yields_zero_psi(self):
        d = DriftDetector(features=["column.T_bot_C", "column.RR"]).fit(_ref_df())
        live = _ref_df(n=200).drop(columns=["column.RR"])
        report = d.score(live)
        rr = next(f for f in report.by_feature if f.feature == "column.RR")
        assert rr.psi == 0.0
        assert rr.status == "none"

    def test_missing_feature_in_reference_is_skipped(self):
        d = DriftDetector(features=["column.T_bot_C", "not_in_ref"]).fit(_ref_df())
        report = d.score(_ref_df(seed=1))
        feats = {r.feature for r in report.by_feature}
        assert "not_in_ref" not in feats
        assert "column.T_bot_C" in feats

    def test_report_to_dict_shape(self):
        d = DriftDetector(features=["column.T_bot_C"]).fit(_ref_df())
        report = d.score(_ref_df(seed=2))
        body = report.to_dict()
        for key in ("overall_status", "max_psi", "worst_feature",
                    "n_live", "by_feature"):
            assert key in body
        assert isinstance(body["by_feature"], list)
        if body["by_feature"]:
            for k in ("feature", "psi", "status", "n_ref", "n_live"):
                assert k in body["by_feature"][0]


# ─────────────────────────────────────────────────────────────────────────────
# /api/drift/status endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def with_loaded(self, monkeypatch):
        from api import server
        ref = _ref_df()
        # Build a fake state.run with enough samples
        live = _ref_df(n=500, seed=2)
        run = MagicMock()
        run.scenario = "test"
        run.process_data = pd.concat([ref, live], ignore_index=True)
        monkeypatch.setattr(server.state, "run", run)
        monkeypatch.setattr(server.state, "replay_idx",
                             len(run.process_data) - 1)
        # Fit a detector on ref only
        detector = DriftDetector(
            features=["column.T_bot_C", "column.RR"]
        ).fit(ref)
        monkeypatch.setattr(server.state, "drift_detector", detector)
        return run

    def test_unavailable_when_detector_not_fitted(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "drift_detector", None)
        body = client.get("/api/drift/status").json()
        assert body["available"] is False

    def test_404_when_no_scenario(self, client, monkeypatch):
        from api import server
        monkeypatch.setattr(server.state, "drift_detector",
                             DriftDetector(features=["x"]).fit(
                                 pd.DataFrame({"x": [1.0, 2.0, 3.0]})))
        monkeypatch.setattr(server.state, "run", None)
        r = client.get("/api/drift/status")
        assert r.status_code == 404

    def test_returns_report_when_loaded(self, client, with_loaded):
        body = client.get("/api/drift/status").json()
        assert body["available"] is True
        assert body["warming_up"] is False
        assert "overall_status" in body
        assert "by_feature" in body

    def test_warming_up_when_window_small(self, client, monkeypatch):
        from api import server
        ref = _ref_df()
        run = MagicMock()
        run.scenario = "test"
        run.process_data = ref
        monkeypatch.setattr(server.state, "run", run)
        monkeypatch.setattr(server.state, "replay_idx", 5)   # only 6 samples
        monkeypatch.setattr(server.state, "drift_detector",
                             DriftDetector(features=["column.T_bot_C"]).fit(ref))
        body = client.get("/api/drift/status").json()
        assert body["warming_up"] is True
