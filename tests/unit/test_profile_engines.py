"""Tests that the AnalyticalEngine and RecommendationEngine consume the
active profile correctly (Bloque U2)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, ".")

from analytics import AnalyticalEngine
from profile import (
    BATCH_PROFILE, PILOT_PROFILE,
    active_profile,
)
from recommendations.engine import RecommendationEngine


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    monkeypatch.delenv("AXION_PROCESS_PROFILE", raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# Profile.load_rules() resolves the rule pack
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadRules:
    def test_pilot_loads_pilot_rules(self):
        rules = PILOT_PROFILE.load_rules()
        assert len(rules) > 0
        # Sanity: all rules have R-prefix names (PILOT_RULES contains R01..R10)
        for r in rules:
            assert r.rule_name.startswith("R")

    def test_batch_loads_batch_rules(self):
        rules = BATCH_PROFILE.load_rules()
        assert len(rules) == 3
        for r in rules:
            assert r.rule_name.startswith("B")

    def test_no_rule_pack_returns_empty(self):
        from profile import ProcessProfile
        p = ProcessProfile(name="x", display_name="X", rule_pack_path=None)
        assert p.load_rules() == []


# ─────────────────────────────────────────────────────────────────────────────
# Profile schema additions
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileSchemaAdditions:
    def test_pilot_has_operational_limits(self):
        assert "cstr.T_R_C" in PILOT_PROFILE.operational_limits

    def test_batch_has_operational_limits(self):
        assert "batch.T_R_C" in BATCH_PROFILE.operational_limits

    def test_pilot_measured_tags_subset_of_tags(self):
        names = set(PILOT_PROFILE.tag_names)
        for m in PILOT_PROFILE.measured_tags:
            assert m in names, f"{m} not in pilot tags"

    def test_batch_measured_tags_subset_of_tags(self):
        names = set(BATCH_PROFILE.tag_names)
        for m in BATCH_PROFILE.measured_tags:
            assert m in names, f"{m} not in batch tags"

    def test_to_dict_round_trip_includes_new_fields(self):
        d = BATCH_PROFILE.to_dict()
        assert "rule_pack_path" in d
        assert "measured_tags" in d


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticalEngine accepts profile tags + operational_limits
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticalEnginePerProfile:
    def _read_csv(self, name: str) -> pd.DataFrame:
        path = Path("data") / f"{name}.csv"
        if not path.exists():
            pytest.skip(f"Missing data CSV: {path}")
        return pd.read_csv(path)

    def test_pilot_engine_fits_pilot_data(self):
        df = self._read_csv("normal")
        engine = AnalyticalEngine(
            tags=PILOT_PROFILE.tag_names,
            operational_limits=PILOT_PROFILE.operational_limits or None,
            training_fraction=1.0,
        )
        engine.fit(df)   # should not raise

    def test_batch_engine_fits_batch_data(self):
        df = self._read_csv("batch_normal")
        engine = AnalyticalEngine(
            tags=BATCH_PROFILE.tag_names,
            operational_limits=BATCH_PROFILE.operational_limits or None,
            training_fraction=1.0,
            warmup_minutes=5.0,
        )
        engine.fit(df)
        # And running it shouldn't blow up
        sessions = engine.run_sessions(df, post_training_only=False)
        assert isinstance(sessions, list)

    def test_pilot_engine_does_not_see_batch_tags(self):
        df = self._read_csv("normal")
        engine = AnalyticalEngine(
            tags=PILOT_PROFILE.tag_names, training_fraction=1.0,
        )
        engine.fit(df)
        for tag in engine.tags:
            assert tag.startswith(("cstr.", "column."))


# ─────────────────────────────────────────────────────────────────────────────
# RecommendationEngine consumes profile rules
# ─────────────────────────────────────────────────────────────────────────────

class TestRecEnginePerProfile:
    def test_pilot_rec_engine_uses_pilot_rules(self):
        re = RecommendationEngine(
            rules=PILOT_PROFILE.load_rules(),
            operational_limits=PILOT_PROFILE.operational_limits,
        )
        names = [r.rule_name for r in re.rules]
        assert any(n.startswith("R") for n in names)
        assert not any(n.startswith("B") for n in names)

    def test_batch_rec_engine_uses_batch_rules(self):
        re = RecommendationEngine(
            rules=BATCH_PROFILE.load_rules(),
            operational_limits=BATCH_PROFILE.operational_limits,
        )
        names = [r.rule_name for r in re.rules]
        assert all(n.startswith("B") for n in names)
        assert len(names) == 3
