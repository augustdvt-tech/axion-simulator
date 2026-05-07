"""Shared pytest fixtures for the Axion AI test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so imports work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run slow tests (LSTM training, NSGA-II runs)"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests skipped unless --run-slow")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="use --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Synthetic process DataFrame (20 rows, no CSV dependency)
# ---------------------------------------------------------------------------

TAGS = [
    "cstr.T_R_C", "cstr.T_J_C", "cstr.C_A", "cstr.F_feed", "cstr.F_cool",
    "cstr.T_feed_C", "cstr.T_cool_in_C", "cstr.P_R", "cstr.conversion",
    "column.x_D", "column.x_B_A", "column.purity_B",
    "column.T_top_C", "column.T_bot_C", "column.RR",
    "column.F_vap_kgh", "column.Q_reb_kW", "column.P_top_bar", "column.P_bot_bar",
]

NOMINAL = {
    "cstr.T_R_C": 78.0, "cstr.T_J_C": 57.0, "cstr.C_A": 172.0,
    "cstr.F_feed": 2.0, "cstr.F_cool": 0.30, "cstr.T_feed_C": 70.0,
    "cstr.T_cool_in_C": 15.0, "cstr.P_R": 2.0, "cstr.conversion": 0.83,
    "column.x_D": 1.0, "column.x_B_A": 0.007, "column.purity_B": 99.3,
    "column.T_top_C": 70.0, "column.T_bot_C": 117.8, "column.RR": 5.5,
    "column.F_vap_kgh": 2150.0, "column.Q_reb_kW": 260.0,
    "column.P_top_bar": 1.02, "column.P_bot_bar": 1.15,
}


@pytest.fixture
def df_synthetic() -> pd.DataFrame:
    """
    60-row synthetic DataFrame at nominal operating conditions with tiny noise.
    60 rows satisfies the PCA minimum (30 train samples at 50% split).
    Safe to use in any unit test — no CSV dependency, runs in < 1ms.
    """
    rng = np.random.default_rng(42)
    n = 60
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
    rows = {
        "timestamp": timestamps,
        "time_s": np.arange(n) * 60.0,
    }
    for tag, nom in NOMINAL.items():
        rows[tag] = rng.normal(nom, abs(nom) * 0.002, n)
    return pd.DataFrame(rows)


@pytest.fixture
def df_with_spike(df_synthetic) -> pd.DataFrame:
    """df_synthetic with a Shewhart-violating spike in cstr.T_R_C at row 50."""
    df = df_synthetic.copy()
    df.loc[50, "cstr.T_R_C"] = NOMINAL["cstr.T_R_C"] + 20.0   # >> 3-sigma
    return df


@pytest.fixture
def df_with_drift(df_synthetic) -> pd.DataFrame:
    """df_synthetic extended to 120 rows with a slow linear drift in cstr.T_R_C."""
    rng = np.random.default_rng(7)
    n = 120
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
    rows = {"timestamp": timestamps, "time_s": np.arange(n) * 60.0}
    for tag, nom in NOMINAL.items():
        signal = np.full(n, nom)
        if tag == "cstr.T_R_C":
            signal = nom + np.linspace(0, 8.0, n)   # 8 degC drift over 2 h
        rows[tag] = signal + rng.normal(0, abs(nom) * 0.001, n)
    return pd.DataFrame(rows)


@pytest.fixture
def df_normal_csv() -> pd.DataFrame:
    """Loads the real normal.csv — use only when you need realistic statistics."""
    path = PROJECT_ROOT / "data" / "normal.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Alert and Session helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def make_alert():
    """Factory for Alert objects. Usage: make_alert(tag='cstr.T_R_C', ...)"""
    from analytics import Alert, AlertType, Severity

    def _make(
        tag="cstr.T_R_C",
        timestamp="2026-01-01 00:10:00",
        detector="SPC",
        alert_type=AlertType.SHEWHART_VIOLATION,
        severity=Severity.HIGH,
        value=95.0,
        limit=81.0,
        message="Test alert",
    ):
        return Alert(
            timestamp=pd.Timestamp(timestamp),
            detector=detector,
            alert_type=alert_type,
            severity=severity,
            tag=tag,
            message=message,
            value=value,
            limit=limit,
        )
    return _make


@pytest.fixture
def simple_sessions(make_alert):
    """Two EventSessions: one SPC thermal, one PCA multivariate."""
    from analytics import group_alerts_into_sessions, AlertType, Severity

    thermal_alerts = [
        make_alert(tag="cstr.T_R_C",
                   timestamp=f"2026-01-01 0{h}:00:00",
                   detector="SPC",
                   alert_type=AlertType.EWMA_VIOLATION,
                   severity=Severity.MEDIUM,
                   value=80.0 + h, limit=81.0)
        for h in range(3)
    ]
    pca_alerts = [
        make_alert(tag=None,
                   timestamp=f"2026-01-01 0{h}:05:00",
                   detector="PCA",
                   alert_type=AlertType.HOTELLING_T2,
                   severity=Severity.HIGH,
                   value=12.0, limit=9.0)
        for h in range(2)
    ]
    return group_alerts_into_sessions(thermal_alerts + pca_alerts, gap_minutes=30.0)


# ---------------------------------------------------------------------------
# Minimal process DataFrame for RecommendationEngine tests
# ---------------------------------------------------------------------------

@pytest.fixture
def df_for_recs(df_synthetic) -> pd.DataFrame:
    """df_synthetic with purity_B pushed below spec to trigger R06."""
    df = df_synthetic.copy()
    df["column.purity_B"] = 97.5   # below spec of 98.5 %
    return df
