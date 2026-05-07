"""Tests for api/uploads.py — CSV validator + endpoint."""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, ".")

from api.uploads import (
    REQUIRED_COLUMNS,
    MIN_ROWS,
    MAX_FILE_BYTES,
    validate_scenario_name,
    validate_csv_bytes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_valid_df(n_rows: int = 120) -> pd.DataFrame:
    """Build a DataFrame with all required columns + numeric data."""
    base = pd.Timestamp("2026-01-01T00:00:00")
    data = {
        "timestamp": [base + pd.Timedelta(minutes=i) for i in range(n_rows)],
    }
    for col in REQUIRED_COLUMNS:
        if col == "timestamp":
            continue
        data[col] = [1.0 + 0.01 * i for i in range(n_rows)]
    return pd.DataFrame(data)


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# validate_scenario_name
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateScenarioName:
    def test_accepts_simple(self):
        assert validate_scenario_name("custom_run") is None

    def test_accepts_digits(self):
        assert validate_scenario_name("run_2026_01") is None

    def test_rejects_empty(self):
        assert validate_scenario_name("") is not None

    def test_rejects_uppercase(self):
        assert validate_scenario_name("Custom") is not None

    def test_rejects_dash(self):
        assert validate_scenario_name("foo-bar") is not None

    def test_rejects_dot(self):
        assert validate_scenario_name("foo.bar") is not None

    def test_rejects_too_short(self):
        assert validate_scenario_name("a") is not None

    def test_rejects_too_long(self):
        assert validate_scenario_name("x" * 41) is not None


# ─────────────────────────────────────────────────────────────────────────────
# validate_csv_bytes — happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCsvBytesValid:
    def test_valid_csv_passes(self):
        result = validate_csv_bytes(_to_csv_bytes(_make_valid_df()))
        assert result.ok is True
        assert result.errors == []

    def test_returns_dataframe(self):
        result = validate_csv_bytes(_to_csv_bytes(_make_valid_df()))
        assert result.df is not None
        assert len(result.df) == 120

    def test_row_count_reported(self):
        result = validate_csv_bytes(_to_csv_bytes(_make_valid_df(n_rows=200)))
        assert result.n_rows == 200


# ─────────────────────────────────────────────────────────────────────────────
# validate_csv_bytes — failure cases
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCsvBytesInvalid:
    def test_empty_bytes_rejected(self):
        result = validate_csv_bytes(b"")
        assert result.ok is False
        assert "Empty" in result.errors[0]

    def test_oversized_rejected(self):
        # craft content larger than MAX_FILE_BYTES
        big = b"a" * (MAX_FILE_BYTES + 1)
        result = validate_csv_bytes(big)
        assert result.ok is False
        assert "too large" in result.errors[0].lower()

    def test_unparseable_csv_rejected(self):
        result = validate_csv_bytes(b"\x00\x01\x02not,a,csv\n\xff\xfe")
        # Either parse error or missing columns — but never ok=True
        assert result.ok is False

    def test_missing_columns_rejected(self):
        df = _make_valid_df()
        df = df.drop(columns=["column.purity_B", "cstr.T_R_C"])
        result = validate_csv_bytes(_to_csv_bytes(df))
        assert result.ok is False
        assert any("Missing required columns" in e for e in result.errors)

    def test_too_few_rows_rejected(self):
        result = validate_csv_bytes(_to_csv_bytes(_make_valid_df(n_rows=10)))
        assert result.ok is False
        assert any("Too few rows" in e for e in result.errors)

    def test_minimum_rows_passes(self):
        result = validate_csv_bytes(_to_csv_bytes(_make_valid_df(n_rows=MIN_ROWS)))
        assert result.ok is True

    def test_non_numeric_column_rejected(self):
        df = _make_valid_df()
        df["cstr.T_R_C"] = "not_a_number"
        result = validate_csv_bytes(_to_csv_bytes(df))
        assert result.ok is False
        assert any("non-numeric" in e for e in result.errors)

    def test_partial_non_numeric_warns(self):
        df = _make_valid_df(n_rows=200)
        # write non-numeric values via raw CSV manipulation (bypass dtype constraint)
        csv_bytes = _to_csv_bytes(df)
        text = csv_bytes.decode()
        lines = text.splitlines()
        header = lines[0].split(",")
        col_idx = header.index("cstr.T_R_C")
        for i in range(1, 31):  # 30 rows = 15%
            cells = lines[i].split(",")
            cells[col_idx] = "x"
            lines[i] = ",".join(cells)
        modified = ("\n".join(lines) + "\n").encode()
        result = validate_csv_bytes(modified)
        assert result.ok is True
        assert any("cstr.T_R_C" in w for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint: POST /api/scenarios/upload
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from api import server
        # Redirect DATA_DIR to a temp folder so we don't pollute data/
        monkeypatch.setattr(server, "DATA_DIR", tmp_path)
        # Stub load_scenario so activation doesn't run the full pipeline
        monkeypatch.setattr(server, "load_scenario",
                            lambda s: type("R", (), {
                                "scenario": s, "process_data": _make_valid_df(),
                                "recommendations": [], "decisions": [],
                                "sessions": [],
                            })())
        return TestClient(server.app, raise_server_exceptions=False)

    def test_rejects_invalid_name(self, client):
        df = _make_valid_df()
        r = client.post(
            "/api/scenarios/upload",
            data={"name": "BadName", "activate": "false"},
            files={"file": ("test.csv", _to_csv_bytes(df), "text/csv")},
        )
        assert r.status_code == 400

    def test_rejects_missing_columns(self, client):
        df = _make_valid_df().drop(columns=["column.purity_B"])
        r = client.post(
            "/api/scenarios/upload",
            data={"name": "custom_run", "activate": "false"},
            files={"file": ("bad.csv", _to_csv_bytes(df), "text/csv")},
        )
        assert r.status_code == 400
        body = r.json()
        # detail can be dict {"errors": [...], "warnings": [...]}
        assert "detail" in body

    def test_accepts_valid_csv_no_activate(self, client, tmp_path):
        df = _make_valid_df()
        r = client.post(
            "/api/scenarios/upload",
            data={"name": "my_upload", "activate": "false"},
            files={"file": ("ok.csv", _to_csv_bytes(df), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["scenario"] == "my_upload"
        assert body["activated"] is False
        assert (tmp_path / "my_upload.csv").exists()

    def test_accepts_valid_csv_with_activate(self, client, tmp_path):
        df = _make_valid_df()
        r = client.post(
            "/api/scenarios/upload",
            data={"name": "my_upload2", "activate": "true"},
            files={"file": ("ok.csv", _to_csv_bytes(df), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["activated"] is True

    def test_response_includes_row_count(self, client):
        df = _make_valid_df(n_rows=150)
        r = client.post(
            "/api/scenarios/upload",
            data={"name": "rows_test", "activate": "false"},
            files={"file": ("ok.csv", _to_csv_bytes(df), "text/csv")},
        )
        assert r.json()["n_rows"] == 150

    def test_overwrites_existing_file(self, client, tmp_path):
        df1 = _make_valid_df(n_rows=100)
        df2 = _make_valid_df(n_rows=200)
        client.post(
            "/api/scenarios/upload",
            data={"name": "ovr", "activate": "false"},
            files={"file": ("a.csv", _to_csv_bytes(df1), "text/csv")},
        )
        r = client.post(
            "/api/scenarios/upload",
            data={"name": "ovr", "activate": "false"},
            files={"file": ("b.csv", _to_csv_bytes(df2), "text/csv")},
        )
        assert r.status_code == 200
        assert r.json()["n_rows"] == 200
