"""
Axion AI - OPC-UA Tag Mapping
=============================

The tag map is the contract between the plant and Axion AI: it declares which
OPC-UA node on which server corresponds to which canonical Axion tag. This
file is the one thing a plant engineer has to edit when deploying Axion AI
to a new site — nothing else in the system is plant-specific.

Tag map YAML/dict structure
---------------------------
{
    "server": {
        "endpoint":       "opc.tcp://192.168.1.50:4840",
        "security_policy": "None",          # or Basic256Sha256
        "username":        "axion",
        "password":        "...",
        "session_timeout_ms": 60000,
    },
    "sampling": {
        "interval_ms":    1000,              # how often to poll
        "aggregation":    "last",            # "last" | "mean"
    },
    "tags": [
        {
            "axion_tag":  "cstr.T_R_C",
            "node_id":    "ns=2;s=Reactor01.Temperature",
            "units":      "C",
            "min_range":  0.0,
            "max_range":  150.0,
            "writable":   False,
        },
        {
            "axion_tag":  "column.RR",
            "node_id":    "ns=2;s=Column01.RR_Setpoint",
            "units":      "dimensionless",
            "min_range":  2.0,
            "max_range":  8.0,
            "writable":   True,     # Axion can write to this node in semi/auto mode
        },
        ...
    ]
}

Why YAML (not code)
-------------------
Tag maps change when instruments are replaced, the process is modified, or a
new Axion deployment targets a different plant. Putting them in code would
force a redeploy for every change. Keeping them in YAML (or JSON) lets a plant
engineer edit with a text editor and reload the service without touching
Python.

The map also carries per-tag metadata (range, units, writability) that Axion
uses for data validation and for deciding which tags are safe targets for
setpoint writes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class TagMapping:
    """A single Axion tag mapped to an OPC-UA node."""
    axion_tag: str
    node_id: str
    units: str = ""
    min_range: Optional[float] = None
    max_range: Optional[float] = None
    writable: bool = False                  # can Axion write setpoints here?
    description: str = ""


@dataclass
class ServerConfig:
    """OPC-UA server connection parameters."""
    endpoint: str                           # opc.tcp://host:port
    security_policy: str = "None"           # "None" | "Basic256Sha256" | ...
    username: Optional[str] = None
    password: Optional[str] = None
    session_timeout_ms: int = 60000
    connect_timeout_s: float = 5.0
    # Optional security material — only consumed when security_policy != "None"
    cert_path:     Optional[str] = None
    key_path:      Optional[str] = None
    security_mode: str = "SignAndEncrypt"   # "Sign" | "SignAndEncrypt"


@dataclass
class SamplingConfig:
    """How to sample the OPC-UA server."""
    interval_ms: int = 1000                 # polling interval
    aggregation: str = "last"               # "last" | "mean"
    batch_size: int = 1                     # samples per batch emitted downstream


@dataclass
class TagMap:
    """Full tag map: server + sampling + list of tags."""
    server: ServerConfig
    sampling: SamplingConfig
    tags: List[TagMapping]

    @property
    def node_id_by_tag(self) -> Dict[str, str]:
        return {t.axion_tag: t.node_id for t in self.tags}

    @property
    def tag_by_node_id(self) -> Dict[str, str]:
        return {t.node_id: t.axion_tag for t in self.tags}

    def writable_tags(self) -> List[TagMapping]:
        return [t for t in self.tags if t.writable]

    @classmethod
    def from_dict(cls, d: dict) -> "TagMap":
        return cls(
            server=ServerConfig(**d.get("server", {})),
            sampling=SamplingConfig(**d.get("sampling", {})),
            tags=[TagMapping(**t) for t in d.get("tags", [])],
        )

    @classmethod
    def from_file(cls, path: Path) -> "TagMap":
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # optional dependency
            except ImportError:
                raise RuntimeError("PyYAML required to parse .yaml tag maps")
            d = yaml.safe_load(text)
        else:
            d = json.loads(text)
        return cls.from_dict(d)


# =============================================================================
# Canonical tag map for the pilot process
# =============================================================================
# This mirrors the simulator's CSV columns. A real plant deployment would
# replace node_id strings with the actual OPC-UA node identifiers from that
# plant's DCS/SCADA. The axion_tag side stays identical — the analytics,
# rules, and consensus layers are unchanged across sites.

PILOT_TAG_MAP = TagMap(
    server=ServerConfig(
        endpoint="opc.tcp://127.0.0.1:4840",
        security_policy="None",
    ),
    sampling=SamplingConfig(interval_ms=1000, aggregation="last"),
    tags=[
        # Reactor (measured)
        TagMapping("cstr.T_R_C",      "ns=2;s=CSTR01.T_R",      units="C",   min_range=0,   max_range=150,  writable=False),
        TagMapping("cstr.T_J_C",      "ns=2;s=CSTR01.T_J",      units="C",   min_range=0,   max_range=100,  writable=False),
        TagMapping("cstr.C_A",        "ns=2;s=CSTR01.C_A",      units="mol/m3", min_range=0, max_range=2000, writable=False),
        TagMapping("cstr.conversion", "ns=2;s=CSTR01.X",        units="-",   min_range=0,   max_range=1,    writable=False),
        TagMapping("cstr.P_R",        "ns=2;s=CSTR01.P_R",      units="bar", min_range=0,   max_range=10,   writable=False),
        TagMapping("cstr.T_feed_C",   "ns=2;s=CSTR01.T_feed",   units="C",   min_range=0,   max_range=150,  writable=False),
        TagMapping("cstr.T_cool_in_C","ns=2;s=CSTR01.T_cool_in",units="C",   min_range=0,   max_range=50,   writable=False),

        # Reactor (manipulated — writable in semi/auto)
        TagMapping("cstr.F_feed",     "ns=2;s=CSTR01.F_feed_SP",units="m3/h", min_range=0, max_range=5,     writable=True),
        TagMapping("cstr.F_cool",     "ns=2;s=CSTR01.F_cool_SP",units="m3/h", min_range=0, max_range=1,     writable=True),

        # Column (measured)
        TagMapping("column.purity_B", "ns=2;s=COL01.PurityB",   units="%",   min_range=0,   max_range=100,  writable=False),
        TagMapping("column.x_D",      "ns=2;s=COL01.x_D",       units="-",   min_range=0,   max_range=1,    writable=False),
        TagMapping("column.x_B_A",    "ns=2;s=COL01.x_B_A",     units="-",   min_range=0,   max_range=1,    writable=False),
        TagMapping("column.T_top_C",  "ns=2;s=COL01.T_top",     units="C",   min_range=0,   max_range=150,  writable=False),
        TagMapping("column.T_bot_C",  "ns=2;s=COL01.T_bot",     units="C",   min_range=0,   max_range=200,  writable=False),
        TagMapping("column.Q_reb_kW", "ns=2;s=COL01.Q_reb",     units="kW",  min_range=0,   max_range=500,  writable=False),
        TagMapping("column.F_vap_kgh","ns=2;s=COL01.F_vap",     units="kg/h",min_range=0,   max_range=2000, writable=False),
        TagMapping("column.P_top_bar","ns=2;s=COL01.P_top",     units="bar", min_range=0,   max_range=5,    writable=False),
        TagMapping("column.P_bot_bar","ns=2;s=COL01.P_bot",     units="bar", min_range=0,   max_range=5,    writable=False),

        # Column (manipulated — writable)
        TagMapping("column.RR",       "ns=2;s=COL01.RR_SP",     units="-",   min_range=2,   max_range=8,    writable=True),
    ],
)
