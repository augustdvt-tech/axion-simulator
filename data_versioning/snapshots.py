"""
Axion AI — Data snapshot store
================================

Content-addressed, git-like versioning for the scenario CSVs in `data/`.
Each snapshot is a manifest pinning every CSV's SHA256 + size + row/col
count at a point in time. Snapshots are written under `data/.versions/`
as JSON, never overwritten.

Why not DVC / lakeFS / S3?
--------------------------
At pilot scale (a handful of CSVs, tens of MB total) the value is in the
*manifest*, not in remote storage. DVC adds operational complexity that
doesn't pay off until the dataset is too large to commit. We use the
exact same content-addressed model so a future migration to DVC is a
straight swap.

Snapshot identity
-----------------
The snapshot_id is the first 12 hex chars of:

    sha256( "\\n".join(f"{name}={sha256}" for name, sha256 in sorted(files)) )

So identical content always produces the same id (free deduplication).
A change to any byte of any CSV produces a different id.

Use cases this enables
----------------------
1. **Reproducibility**: every trained model records the snapshot id of the
   data it saw, so we can answer "what data was this model trained on?"
2. **Drift alarm**: `verify_snapshot()` re-hashes the current files and
   reports any divergence from the snapshot — useful in CI to catch
   accidental edits to the training corpus.
3. **Audit**: `make data-snapshot --message "before adding sensor_failure"`
   stamps a checkpoint with intent.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_HASH_CHUNK = 64 * 1024


@dataclass(frozen=True)
class FileMeta:
    name:        str       # e.g. "normal.csv"
    sha256:      str
    size_bytes:  int
    n_rows:      int        # CSV rows excluding header (-1 if not parseable)
    n_cols:      int
    modified_at: str        # ISO 8601 UTC
    relpath:     str = ""   # path relative to data_dir; defaults to name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileMeta":
        return cls(
            name=d["name"],
            sha256=d["sha256"],
            size_bytes=int(d.get("size_bytes", 0)),
            n_rows=int(d.get("n_rows", -1)),
            n_cols=int(d.get("n_cols", 0)),
            modified_at=d.get("modified_at", ""),
            relpath=d.get("relpath", d["name"]),
        )


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    created_at:  str         # ISO 8601 UTC
    message:     str
    files:       List[FileMeta] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at":  self.created_at,
            "message":     self.message,
            "files":       [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Snapshot":
        return cls(
            snapshot_id=d["snapshot_id"],
            created_at=d["created_at"],
            message=d.get("message", ""),
            files=[FileMeta.from_dict(f) for f in d.get("files", [])],
        )

    def file_index(self) -> Dict[str, FileMeta]:
        return {f.name: f for f in self.files}


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _csv_dimensions(path: Path) -> Tuple[int, int]:
    """Return (n_rows excluding header, n_cols) without loading the full file
    into memory. Returns (-1, 0) if the file isn't parseable as CSV."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
            if not header:
                return 0, 0
            n_cols = len(header.rstrip("\r\n").split(","))
            n_rows = sum(1 for _ in fh)
        return n_rows, n_cols
    except Exception:
        return -1, 0


def compute_file_meta(path: Path, data_dir: Optional[Path] = None) -> FileMeta:
    """Compute metadata for a single file. `data_dir` controls relpath."""
    p = Path(path)
    sha = _sha256_file(p)
    size = p.stat().st_size
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds")
    n_rows, n_cols = _csv_dimensions(p)
    relpath = str(p.name) if data_dir is None else str(p.relative_to(data_dir))
    return FileMeta(
        name=p.name, sha256=sha, size_bytes=size,
        n_rows=n_rows, n_cols=n_cols,
        modified_at=mtime, relpath=relpath,
    )


def compute_snapshot_id(files: Iterable[FileMeta]) -> str:
    """Deterministic, content-addressed id derived from file sha256s."""
    parts = sorted((f.name, f.sha256) for f in files)
    payload = "\n".join(f"{name}={sha}" for name, sha in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot store I/O
# ─────────────────────────────────────────────────────────────────────────────

def take_snapshot(
    data_dir: Path,
    snapshots_dir: Path,
    message: str = "",
    pattern: str = "*.csv",
) -> Snapshot:
    """Hash every file in `data_dir` matching `pattern`, write a manifest,
    return the Snapshot. Idempotent on content — calling twice on the same
    files yields the same snapshot_id and overwrites the manifest with the
    new message/timestamp.
    """
    data_dir = Path(data_dir)
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    files: List[FileMeta] = []
    for p in sorted(data_dir.glob(pattern)):
        if p.name.startswith("."):
            continue
        files.append(compute_file_meta(p, data_dir=data_dir))

    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found under {data_dir}"
        )

    snap_id = compute_snapshot_id(files)
    snapshot = Snapshot(
        snapshot_id=snap_id,
        created_at=_now_iso(),
        message=message,
        files=files,
    )
    manifest_path = snapshots_dir / f"{snap_id}.json"
    manifest_path.write_text(json.dumps(snapshot.to_dict(), indent=2))
    return snapshot


def load_snapshot(snapshots_dir: Path, snapshot_id: str) -> Snapshot:
    path = Path(snapshots_dir) / f"{snapshot_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot {snapshot_id!r} not found at {path}")
    return Snapshot.from_dict(json.loads(path.read_text()))


def list_snapshots(snapshots_dir: Path) -> List[Snapshot]:
    """Return every snapshot manifest in chronological order (oldest first)."""
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.exists():
        return []
    out: List[Snapshot] = []
    for p in sorted(snapshots_dir.glob("*.json")):
        try:
            out.append(Snapshot.from_dict(json.loads(p.read_text())))
        except Exception:
            continue
    out.sort(key=lambda s: s.created_at)
    return out


def diff_snapshots(a: Snapshot, b: Snapshot) -> Dict[str, List[Dict[str, Any]]]:
    """Compare two snapshots. Returns {added, removed, changed} where each
    entry is a dict with file metadata."""
    idx_a = a.file_index()
    idx_b = b.file_index()

    added   = [idx_b[n].to_dict() for n in sorted(idx_b) if n not in idx_a]
    removed = [idx_a[n].to_dict() for n in sorted(idx_a) if n not in idx_b]
    changed: List[Dict[str, Any]] = []
    for n in sorted(set(idx_a) & set(idx_b)):
        if idx_a[n].sha256 != idx_b[n].sha256:
            changed.append({
                "name":        n,
                "sha256_a":    idx_a[n].sha256,
                "sha256_b":    idx_b[n].sha256,
                "n_rows_a":    idx_a[n].n_rows,
                "n_rows_b":    idx_b[n].n_rows,
            })
    return {"added": added, "removed": removed, "changed": changed}


def verify_snapshot(
    snapshot: Snapshot, data_dir: Path,
) -> Dict[str, Any]:
    """Re-hash files in `data_dir` against the snapshot's manifest. Returns
    {ok: bool, missing, extra, mismatched}. `ok` is True iff the current
    state of `data_dir` exactly matches the snapshot."""
    data_dir = Path(data_dir)
    expected = snapshot.file_index()
    actual_paths = {p.name: p for p in data_dir.glob("*.csv")
                    if not p.name.startswith(".")}

    missing    = [name for name in sorted(expected) if name not in actual_paths]
    extra      = [name for name in sorted(actual_paths) if name not in expected]
    mismatched: List[Dict[str, Any]] = []
    for name in sorted(set(expected) & set(actual_paths)):
        cur = compute_file_meta(actual_paths[name], data_dir=data_dir)
        if cur.sha256 != expected[name].sha256:
            mismatched.append({
                "name":     name,
                "expected": expected[name].sha256,
                "actual":   cur.sha256,
            })

    return {
        "ok": not missing and not extra and not mismatched,
        "missing":    missing,
        "extra":      extra,
        "mismatched": mismatched,
    }
