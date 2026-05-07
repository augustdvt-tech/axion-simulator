"""Tests for profile/ — ProcessProfile, registry, batch reactor sim, endpoints."""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from profile import (
    BATCH_PROFILE, PILOT_PROFILE, ProcessProfile, TagSpec,
    active_profile, active_profile_name,
    get_profile, list_profiles, register,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    monkeypatch.delenv("AXION_PROCESS_PROFILE", raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# ProcessProfile schema
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessProfileSchema:
    def test_tag_names_property(self):
        p = ProcessProfile(
            name="x", display_name="X",
            tags=[TagSpec("a.x", "A"), TagSpec("a.y", "B")],
        )
        assert p.tag_names == ["a.x", "a.y"]

    def test_live_columns_includes_timestamp_first(self):
        p = ProcessProfile(name="x", display_name="X",
                            tags=[TagSpec("a.x", "A")])
        assert p.live_columns[0] == "timestamp"
        assert p.live_columns[1:] == ["a.x"]

    def test_kpi_tags_filters_out_non_kpi(self):
        p = ProcessProfile(name="x", display_name="X", tags=[
            TagSpec("k.1", "K1", is_kpi=True),
            TagSpec("k.2", "K2", is_kpi=False),
        ])
        assert [t.tag for t in p.kpi_tags] == ["k.1"]

    def test_tag_lookup_returns_none_for_unknown(self):
        p = ProcessProfile(name="x", display_name="X",
                            tags=[TagSpec("a.x", "A")])
        assert p.tag("a.x") is not None
        assert p.tag("nope") is None

    def test_to_dict_round_trippable_keys(self):
        p = ProcessProfile(name="x", display_name="X",
                            tags=[TagSpec("a.x", "A", "°C", 0.0, 100.0)])
        d = p.to_dict()
        for key in ("name", "display_name", "tags", "feature_cols",
                    "target_col", "scenarios", "purity_kpi",
                    "purity_spec_min"):
            assert key in d
        assert d["tags"][0]["spec_min"] == 0.0
        assert d["tags"][0]["spec_max"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Registry / env
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_pilot_and_batch_registered(self):
        names = list_profiles()
        assert "pilot" in names
        assert "batch_reactor" in names

    def test_get_profile_returns_known(self):
        assert get_profile("pilot") is PILOT_PROFILE

    def test_get_profile_raises_for_unknown(self):
        with pytest.raises(KeyError):
            get_profile("ghost")

    def test_register_adds_new_profile(self):
        custom = ProcessProfile(name="test_custom_x", display_name="X",
                                  tags=[TagSpec("x.a", "A")])
        register(custom)
        try:
            assert "test_custom_x" in list_profiles()
            assert get_profile("test_custom_x") is custom
        finally:
            from profile.process_profile import _REGISTRY
            _REGISTRY.pop("test_custom_x", None)

    def test_active_profile_defaults_to_pilot(self):
        assert active_profile_name() == "pilot"
        assert active_profile() is PILOT_PROFILE

    def test_active_profile_respects_env(self, monkeypatch):
        monkeypatch.setenv("AXION_PROCESS_PROFILE", "batch_reactor")
        assert active_profile_name() == "batch_reactor"
        assert active_profile() is BATCH_PROFILE


# ─────────────────────────────────────────────────────────────────────────────
# Concrete profiles — sanity checks on contents
# ─────────────────────────────────────────────────────────────────────────────

class TestConcreteProfiles:
    def test_pilot_has_purity_kpi(self):
        assert PILOT_PROFILE.purity_kpi == "column.purity_B"
        assert PILOT_PROFILE.purity_spec_min == 98.5

    def test_pilot_lists_known_scenarios(self):
        for s in ("normal", "thermal_drift", "feed_perturbation"):
            assert s in PILOT_PROFILE.scenarios

    def test_batch_has_purity_kpi(self):
        assert BATCH_PROFILE.purity_kpi == "batch.conversion"
        assert BATCH_PROFILE.purity_spec_min == 0.85

    def test_batch_tags_use_batch_prefix(self):
        for t in BATCH_PROFILE.tags:
            assert t.tag.startswith("batch.")

    def test_pilot_and_batch_disjoint_namespace(self):
        pilot_tags = set(PILOT_PROFILE.tag_names)
        batch_tags = set(BATCH_PROFILE.tag_names)
        assert pilot_tags.isdisjoint(batch_tags)


# ─────────────────────────────────────────────────────────────────────────────
# Batch reactor simulator
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchReactorSim:
    @pytest.fixture(scope="class")
    def df(self):
        from simulator.batch_reactor import BatchParams, simulate_batch
        # Short sim for speed (1 hour)
        return simulate_batch(BatchParams(duration_min=60.0))

    def test_returns_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_canonical_columns(self, df):
        for col in BATCH_PROFILE.live_columns:
            assert col in df.columns

    def test_temperature_increases_then_levels_off(self, df):
        # Exothermic reaction → temperature should rise from initial
        T_start = df["batch.T_R_C"].iloc[0]
        T_max   = df["batch.T_R_C"].max()
        assert T_max > T_start

    def test_reactant_concentration_decreases_monotonically(self, df):
        c_a = df["batch.C_A"].values
        # Allow tiny numerical noise; require global decrease
        assert c_a[-1] < c_a[0]
        # Check no big rebound (numerical robustness)
        assert (np.diff(c_a) <= 1e-6).all()

    def test_conversion_increases_monotonically(self, df):
        conv = df["batch.conversion"].values
        assert (np.diff(conv) >= -1e-9).all()
        assert conv[0] == pytest.approx(0.0, abs=1e-9)
        assert conv[-1] > 0

    def test_runaway_scenario_higher_peak_temp(self):
        from simulator.batch_reactor import BatchParams, simulate_batch
        normal  = simulate_batch(BatchParams(duration_min=120.0))
        runaway = simulate_batch(BatchParams(
            duration_min=120.0,
            F_cool_schedule={0.0: 0.4, 60.0: 0.05},
        ))
        assert runaway["batch.T_R_C"].max() > normal["batch.T_R_C"].max()


# ─────────────────────────────────────────────────────────────────────────────
# Profile-aware report
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileAwareReport:
    def test_pilot_kpi_summary_uses_pilot_columns(self):
        from api.report import kpi_summary
        df = pd.DataFrame({
            "timestamp":       pd.date_range("2026-01-01", periods=10, freq="1min"),
            "column.purity_B": [99.0] * 10,
            "cstr.T_R_C":      [78.0] * 10,
        })
        result = kpi_summary(df, profile=PILOT_PROFILE)
        cols = {row["col"] for row in result["rows"]}
        assert "column.purity_B" in cols
        assert "cstr.T_R_C" in cols

    def test_batch_kpi_summary_uses_batch_columns(self):
        from api.report import kpi_summary
        df = pd.DataFrame({
            "timestamp":        pd.date_range("2026-01-01", periods=10, freq="1min"),
            "batch.T_R_C":      [60.0] * 10,
            "batch.conversion": [0.5]  * 10,
        })
        result = kpi_summary(df, profile=BATCH_PROFILE)
        cols = {row["col"] for row in result["rows"]}
        assert "batch.T_R_C" in cols
        assert "batch.conversion" in cols
        # No pilot columns leak in
        assert "column.purity_B" not in cols

    def test_batch_purity_below_spec_uses_conversion(self):
        from api.report import kpi_summary
        df = pd.DataFrame({
            "timestamp":        pd.date_range("2026-01-01", periods=10, freq="1min"),
            "batch.conversion": [0.50] * 5 + [0.90] * 5,   # half below 0.85 spec
        })
        result = kpi_summary(df, profile=BATCH_PROFILE)
        assert result["purity_below_spec_pct"] == pytest.approx(50.0)

    def test_render_html_with_batch_profile(self):
        from api.report import (
            kpi_summary, recommendations_summary, decisions_summary,
            sessions_summary, render_html,
        )
        df = pd.DataFrame({
            "timestamp":        pd.date_range("2026-01-01", periods=10, freq="1min"),
            "batch.conversion": [0.9] * 10,
        })
        html = render_html(
            scenario="batch_normal",
            generated_at="2026-01-01T00:00:00",
            kpi=kpi_summary(df, profile=BATCH_PROFILE),
            rec_summary=recommendations_summary([]),
            dec_summary=decisions_summary([]),
            sess_summary=sessions_summary([]),
            perf_rows=[], rec_log=[],
            profile=BATCH_PROFILE,
        )
        # The batch profile's purity tag is "batch.conversion" labeled
        # "Batch Conversion" — the executive tile should reflect that
        assert "Batch Conversion" in html
        # And no leakage of pilot product purity label
        assert "Product Purity" not in html


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints: /api/profile and /api/profile/select
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app, raise_server_exceptions=False)

    def test_get_profile_returns_active(self, client):
        body = client.get("/api/profile").json()
        assert body["active"] in list_profiles()
        assert "profile" in body
        assert "tags" in body["profile"]

    def test_get_profile_includes_available_list(self, client):
        body = client.get("/api/profile").json()
        assert "pilot" in body["available"]
        assert "batch_reactor" in body["available"]

    def test_select_profile_400_on_unknown(self, client):
        r = client.post("/api/profile/select", json={"profile": "ghost"})
        assert r.status_code == 400

    def test_select_profile_400_on_empty(self, client):
        r = client.post("/api/profile/select", json={})
        assert r.status_code == 400

    def test_select_profile_switches_active(self, client, monkeypatch):
        try:
            r = client.post("/api/profile/select",
                            json={"profile": "batch_reactor"})
            assert r.status_code == 200
            assert r.json()["profile"]["name"] == "batch_reactor"
            # Subsequent GET reflects the switch
            assert client.get("/api/profile").json()["active"] == "batch_reactor"
        finally:
            # Restore for downstream tests
            import os
            os.environ.pop("AXION_PROCESS_PROFILE", None)
