"""Tests for data_versioning/ + the /api/data/snapshots endpoints (Bloque Z)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from data_versioning import (
    Snapshot, FileMeta,
    compute_file_meta, compute_snapshot_id,
    diff_snapshots, list_snapshots,
    load_snapshot, take_snapshot, verify_snapshot,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, n_rows: int = 10, n_cols: int = 3) -> None:
    header = ",".join(f"c{i}" for i in range(n_cols))
    rows = [",".join(str(j + i) for j in range(n_cols)) for i in range(n_rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


@pytest.fixture
def data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    _write_csv(d / "normal.csv",       n_rows=20, n_cols=4)
    _write_csv(d / "thermal_drift.csv", n_rows=15, n_cols=4)
    return d


@pytest.fixture
def snapshots_dir(tmp_path) -> Path:
    return tmp_path / "data" / ".versions"


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFileMeta:
    def test_meta_includes_required_fields(self, data_dir):
        meta = compute_file_meta(data_dir / "normal.csv", data_dir=data_dir)
        for key in ("name", "sha256", "size_bytes", "n_rows", "n_cols",
                    "modified_at", "relpath"):
            assert hasattr(meta, key)

    def test_sha256_is_hex(self, data_dir):
        meta = compute_file_meta(data_dir / "normal.csv")
        assert len(meta.sha256) == 64
        int(meta.sha256, 16)   # parses as hex

    def test_n_rows_excludes_header(self, data_dir):
        meta = compute_file_meta(data_dir / "normal.csv")
        assert meta.n_rows == 20   # we wrote 20 data rows

    def test_n_cols_correct(self, data_dir):
        meta = compute_file_meta(data_dir / "normal.csv")
        assert meta.n_cols == 4

    def test_relpath_defaults_to_name(self, data_dir):
        meta = compute_file_meta(data_dir / "normal.csv")
        assert meta.relpath == "normal.csv"

    def test_handles_unparseable_csv(self, tmp_path):
        p = tmp_path / "binary.csv"
        p.write_bytes(b"")   # empty
        meta = compute_file_meta(p)
        assert meta.n_rows == 0


class TestComputeSnapshotId:
    def test_deterministic(self, data_dir):
        snap1 = take_snapshot(data_dir, data_dir / ".v")
        snap2 = take_snapshot(data_dir, data_dir / ".v")
        assert snap1.snapshot_id == snap2.snapshot_id

    def test_changes_when_content_changes(self, data_dir):
        snap1 = take_snapshot(data_dir, data_dir / ".v")
        # Mutate one file
        _write_csv(data_dir / "normal.csv", n_rows=999, n_cols=4)
        snap2 = take_snapshot(data_dir, data_dir / ".v")
        assert snap1.snapshot_id != snap2.snapshot_id

    def test_independent_of_message(self, data_dir):
        a = take_snapshot(data_dir, data_dir / ".v", message="A")
        b = take_snapshot(data_dir, data_dir / ".v", message="B")
        assert a.snapshot_id == b.snapshot_id   # content is identical

    def test_id_length(self):
        sid = compute_snapshot_id([
            FileMeta(name="x", sha256="a"*64, size_bytes=0, n_rows=0,
                       n_cols=0, modified_at=""),
        ])
        assert len(sid) == 12


# ─────────────────────────────────────────────────────────────────────────────
# take_snapshot / load_snapshot / list_snapshots
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotIO:
    def test_take_writes_manifest(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir, message="test")
        manifest = snapshots_dir / f"{snap.snapshot_id}.json"
        assert manifest.exists()
        body = json.loads(manifest.read_text())
        assert body["snapshot_id"] == snap.snapshot_id

    def test_load_round_trip(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir, message="round-trip")
        loaded = load_snapshot(snapshots_dir, snap.snapshot_id)
        assert loaded.snapshot_id == snap.snapshot_id
        assert loaded.message == "round-trip"
        assert len(loaded.files) == len(snap.files)

    def test_load_missing_raises(self, snapshots_dir):
        with pytest.raises(FileNotFoundError):
            load_snapshot(snapshots_dir, "nope")

    def test_take_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            take_snapshot(tmp_path / "empty", tmp_path / ".v")

    def test_list_returns_all_snapshots(self, data_dir, snapshots_dir):
        a = take_snapshot(data_dir, snapshots_dir, message="A")
        # Mutate to make a different snapshot
        _write_csv(data_dir / "thermal_drift.csv", n_rows=99)
        b = take_snapshot(data_dir, snapshots_dir, message="B")
        ids = {s.snapshot_id for s in list_snapshots(snapshots_dir)}
        assert {a.snapshot_id, b.snapshot_id} <= ids

    def test_list_empty_when_no_snapshots(self, tmp_path):
        assert list_snapshots(tmp_path / "noexist") == []


# ─────────────────────────────────────────────────────────────────────────────
# diff_snapshots
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffSnapshots:
    def test_no_diff_for_identical(self, data_dir, snapshots_dir):
        a = take_snapshot(data_dir, snapshots_dir)
        b = take_snapshot(data_dir, snapshots_dir)
        d = diff_snapshots(a, b)
        assert d["added"] == [] and d["removed"] == [] and d["changed"] == []

    def test_detects_added_file(self, data_dir, snapshots_dir):
        a = take_snapshot(data_dir, snapshots_dir)
        _write_csv(data_dir / "new.csv", n_rows=5)
        b = take_snapshot(data_dir, snapshots_dir)
        d = diff_snapshots(a, b)
        assert any(f["name"] == "new.csv" for f in d["added"])

    def test_detects_removed_file(self, data_dir, snapshots_dir):
        a = take_snapshot(data_dir, snapshots_dir)
        (data_dir / "thermal_drift.csv").unlink()
        b = take_snapshot(data_dir, snapshots_dir)
        d = diff_snapshots(a, b)
        assert any(f["name"] == "thermal_drift.csv" for f in d["removed"])

    def test_detects_changed_file(self, data_dir, snapshots_dir):
        a = take_snapshot(data_dir, snapshots_dir)
        _write_csv(data_dir / "normal.csv", n_rows=99)
        b = take_snapshot(data_dir, snapshots_dir)
        d = diff_snapshots(a, b)
        assert any(f["name"] == "normal.csv" for f in d["changed"])
        c = next(f for f in d["changed"] if f["name"] == "normal.csv")
        assert c["sha256_a"] != c["sha256_b"]


# ─────────────────────────────────────────────────────────────────────────────
# verify_snapshot
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifySnapshot:
    def test_ok_when_unchanged(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir)
        result = verify_snapshot(snap, data_dir)
        assert result["ok"] is True

    def test_detects_modified_file(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir)
        _write_csv(data_dir / "normal.csv", n_rows=999)
        result = verify_snapshot(snap, data_dir)
        assert result["ok"] is False
        assert any(m["name"] == "normal.csv" for m in result["mismatched"])

    def test_detects_missing_file(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir)
        (data_dir / "thermal_drift.csv").unlink()
        result = verify_snapshot(snap, data_dir)
        assert result["ok"] is False
        assert "thermal_drift.csv" in result["missing"]

    def test_detects_extra_file(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir)
        _write_csv(data_dir / "extra.csv", n_rows=3)
        result = verify_snapshot(snap, data_dir)
        assert result["ok"] is False
        assert "extra.csv" in result["extra"]


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot dataclass round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotDataclass:
    def test_to_dict_from_dict_round_trip(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir, message="test")
        d = snap.to_dict()
        snap2 = Snapshot.from_dict(d)
        assert snap2.snapshot_id == snap.snapshot_id
        assert snap2.message == snap.message
        assert len(snap2.files) == len(snap.files)

    def test_file_index_keyed_by_name(self, data_dir, snapshots_dir):
        snap = take_snapshot(data_dir, snapshots_dir)
        idx = snap.file_index()
        assert "normal.csv" in idx
        assert idx["normal.csv"].n_rows == 20


# ─────────────────────────────────────────────────────────────────────────────
# /api/data/snapshots endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotEndpoints:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from api import server as srv
        # Use a temp data dir so we don't touch the real one
        d = tmp_path / "data"
        _write_csv(d / "normal.csv",       n_rows=20, n_cols=4)
        _write_csv(d / "thermal_drift.csv", n_rows=15, n_cols=4)
        monkeypatch.setattr(srv, "DATA_DIR", d)
        monkeypatch.setattr(srv, "SNAPSHOTS_DIR", d / ".versions")
        return TestClient(srv.app, raise_server_exceptions=False)

    def test_list_empty_initially(self, client):
        body = client.get("/api/data/snapshots").json()
        assert body["count"] == 0
        assert body["snapshots"] == []

    def test_list_after_taking_snapshot(self, client, tmp_path):
        from data_versioning import take_snapshot
        d = tmp_path / "data"
        snap = take_snapshot(d, d / ".versions", message="e2e")
        body = client.get("/api/data/snapshots").json()
        assert body["count"] >= 1
        assert any(s["snapshot_id"] == snap.snapshot_id
                   for s in body["snapshots"])

    def test_get_snapshot_returns_manifest(self, client, tmp_path):
        from data_versioning import take_snapshot
        d = tmp_path / "data"
        snap = take_snapshot(d, d / ".versions")
        body = client.get(f"/api/data/snapshots/{snap.snapshot_id}").json()
        assert body["snapshot_id"] == snap.snapshot_id
        assert isinstance(body["files"], list)

    def test_get_unknown_snapshot_returns_404(self, client):
        r = client.get("/api/data/snapshots/deadbeef")
        assert r.status_code == 404

    def test_verify_returns_ok_when_unchanged(self, client, tmp_path):
        from data_versioning import take_snapshot
        d = tmp_path / "data"
        snap = take_snapshot(d, d / ".versions")
        body = client.get(f"/api/data/snapshots/{snap.snapshot_id}/verify").json()
        assert body["ok"] is True

    def test_verify_detects_modification(self, client, tmp_path):
        from data_versioning import take_snapshot
        d = tmp_path / "data"
        snap = take_snapshot(d, d / ".versions")
        _write_csv(d / "normal.csv", n_rows=999, n_cols=4)
        body = client.get(f"/api/data/snapshots/{snap.snapshot_id}/verify").json()
        assert body["ok"] is False

    def test_verify_404_for_unknown_snapshot(self, client):
        r = client.get("/api/data/snapshots/deadbeef/verify")
        assert r.status_code == 404
