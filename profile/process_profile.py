"""
Axion AI — Process profile abstraction
=======================================

A `ProcessProfile` is a declarative bundle of the per-process knowledge that
the rest of the platform needs to host a new process: tag list, KPI
definitions, expected ranges, ML feature/target columns, list of available
scenarios, default `live` data buffer columns, display name.

The pilot process (CSTR + binary distillation column) is captured as
`PILOT_PROFILE`. A second profile (`BATCH_PROFILE`) is provided to
demonstrate that the platform is genuinely multi-process — its own tags,
KPIs and scenarios live alongside the pilot's, and either can be made the
active profile via env or `/api/profile/select`.

Out of scope for this block (U)
-------------------------------
The recommendation rules R1–R10 and the analytics engine are still pinned
to the pilot tag schema. Loading a non-pilot profile lets you ingest CSVs,
view KPI tables, generate reports, and run the dashboard — but the rule
engine will only fire on pilot tags. Building rule packs per profile is
the natural follow-up (block "U2 — Rule packs").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TagSpec:
    """One process variable's metadata. Mirrors the per-tag config that lived
    in `_KPI_DEFS` (api/report.py) plus the OPC-UA ranges from `tag_map.py`."""
    tag:        str
    label:      str
    units:      str  = ""
    spec_min:   Optional[float] = None
    spec_max:   Optional[float] = None
    is_kpi:     bool = True
    description: str = ""


@dataclass(frozen=True)
class ProcessProfile:
    """Everything the platform needs to host a process."""
    name:          str           # canonical id ("pilot", "batch_reactor")
    display_name:  str           # human label for the UI
    description:   str = ""
    tags:          List[TagSpec] = field(default_factory=list)
    feature_cols:  List[str] = field(default_factory=list)   # for soft sensor
    target_col:    Optional[str] = None
    scenarios:     List[str] = field(default_factory=list)   # known scenario names
    purity_kpi:    Optional[str] = None                       # tag whose spec_min defines "below spec"
    purity_spec_min: float = 98.5
    # Tags considered "physical measurements" — the FrozenSensorDetector
    # only watches these. Manipulated/setpoint variables are excluded.
    measured_tags: List[str] = field(default_factory=list)
    # Operational limits per tag for the TrendDetector. Shape:
    #   {tag: {"low": float|None, "high": float|None, "rate_per_min": float|None}}
    operational_limits: Dict = field(default_factory=dict)
    # Builder for the rule pack used by RecommendationEngine. Returns a
    # list of DiagnosticRule. Done as a name path rather than imported
    # callable so the profile module stays free of circular imports.
    rule_pack_path: Optional[str] = None    # e.g. "recommendations.rules_pilot:PILOT_RULES"

    # Convenience views
    @property
    def tag_names(self) -> List[str]:
        return [t.tag for t in self.tags]

    @property
    def live_columns(self) -> List[str]:
        """Canonical column shape for live data (timestamp + tags)."""
        return ["timestamp"] + self.tag_names

    @property
    def kpi_tags(self) -> List[TagSpec]:
        return [t for t in self.tags if t.is_kpi]

    def tag(self, name: str) -> Optional[TagSpec]:
        for t in self.tags:
            if t.tag == name:
                return t
        return None

    def load_rules(self) -> List:
        """Resolve `rule_pack_path` to the actual list of DiagnosticRule.

        Returns [] when no rule pack is declared. Done at call time (not
        import time) to keep the profile module decoupled from the rules
        package, so importing `profile/` doesn't drag in pandas/recs.
        """
        if not self.rule_pack_path:
            return []
        import importlib
        mod_name, attr = self.rule_pack_path.split(":")
        mod = importlib.import_module(mod_name)
        return list(getattr(mod, attr))

    def to_dict(self) -> Dict:
        return {
            "name":         self.name,
            "display_name": self.display_name,
            "description":  self.description,
            "feature_cols": list(self.feature_cols),
            "target_col":   self.target_col,
            "scenarios":    list(self.scenarios),
            "purity_kpi":   self.purity_kpi,
            "purity_spec_min": self.purity_spec_min,
            "rule_pack_path":  self.rule_pack_path,
            "measured_tags":   list(self.measured_tags),
            "tags": [
                {
                    "tag":      t.tag,
                    "label":    t.label,
                    "units":    t.units,
                    "spec_min": t.spec_min,
                    "spec_max": t.spec_max,
                    "is_kpi":   t.is_kpi,
                    "description": t.description,
                }
                for t in self.tags
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, ProcessProfile] = {}


def register(profile: ProcessProfile) -> None:
    _REGISTRY[profile.name] = profile


def get_profile(name: str) -> ProcessProfile:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown process profile: {name!r}. "
                       f"Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_profiles() -> List[str]:
    return sorted(_REGISTRY.keys())


def active_profile_name(default: str = "pilot") -> str:
    """Return the active profile id from env or the default."""
    return os.environ.get("AXION_PROCESS_PROFILE", "").strip() or default


def active_profile(default: str = "pilot") -> ProcessProfile:
    return get_profile(active_profile_name(default))
