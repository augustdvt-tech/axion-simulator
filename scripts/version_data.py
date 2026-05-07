"""
Axion AI — Data versioning CLI
================================

Snapshot, list, diff and verify the scenario CSVs in data/. Snapshots
are written to data/.versions/<snapshot_id>.json and are content-addressed
(identical content always produces the same id).

Examples:
    python scripts/version_data.py snapshot --message "added sensor_failure"
    python scripts/version_data.py list
    python scripts/version_data.py show <id>
    python scripts/version_data.py diff <id_a> <id_b>
    python scripts/version_data.py verify <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_versioning import (
    diff_snapshots, list_snapshots, load_snapshot,
    take_snapshot, verify_snapshot,
)


PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
SNAPSHOTS_DIR = DATA_DIR / ".versions"


def cmd_snapshot(args) -> None:
    snap = take_snapshot(DATA_DIR, SNAPSHOTS_DIR, message=args.message or "")
    print(f"  Snapshot {snap.snapshot_id} — {len(snap.files)} files")
    if snap.message:
        print(f"  Message:  {snap.message}")
    print(f"  Manifest: {SNAPSHOTS_DIR / (snap.snapshot_id + '.json')}")


def cmd_list(args) -> None:
    snaps = list_snapshots(SNAPSHOTS_DIR)
    if not snaps:
        print("(no snapshots yet — run `make data-snapshot`)")
        return
    print(f"{'ID':14}  {'CREATED':25s}  {'FILES':>5}  MESSAGE")
    print("-" * 80)
    for s in snaps:
        msg = (s.message[:38] + "…") if len(s.message) > 39 else s.message
        print(f"{s.snapshot_id:14}  {s.created_at:25s}  "
              f"{len(s.files):>5}  {msg}")


def cmd_show(args) -> None:
    snap = load_snapshot(SNAPSHOTS_DIR, args.id)
    print(f"Snapshot:   {snap.snapshot_id}")
    print(f"Created:    {snap.created_at}")
    print(f"Message:    {snap.message or '(none)'}")
    print(f"Files:      {len(snap.files)}")
    print()
    print(f"  {'NAME':30s}  {'SHA256':14s}  {'ROWS':>7s}  {'COLS':>5s}  SIZE")
    for f in snap.files:
        print(f"  {f.name:30s}  {f.sha256[:14]}  {f.n_rows:>7d}  "
              f"{f.n_cols:>5d}  {f.size_bytes}")


def cmd_diff(args) -> None:
    a = load_snapshot(SNAPSHOTS_DIR, args.a)
    b = load_snapshot(SNAPSHOTS_DIR, args.b)
    d = diff_snapshots(a, b)
    print(f"Diff {a.snapshot_id} → {b.snapshot_id}")
    print(f"  Added:     {len(d['added'])}")
    print(f"  Removed:   {len(d['removed'])}")
    print(f"  Changed:   {len(d['changed'])}")
    for f in d["added"]:
        print(f"    + {f['name']}")
    for f in d["removed"]:
        print(f"    - {f['name']}")
    for f in d["changed"]:
        print(f"    ~ {f['name']}  "
              f"({f['sha256_a'][:10]} → {f['sha256_b'][:10]}, "
              f"rows {f['n_rows_a']} → {f['n_rows_b']})")


def cmd_verify(args) -> None:
    snap = load_snapshot(SNAPSHOTS_DIR, args.id)
    result = verify_snapshot(snap, DATA_DIR)
    print(f"Snapshot:  {snap.snapshot_id}")
    print(f"Status:    {'OK ✓' if result['ok'] else 'DIVERGED ✗'}")
    if result["missing"]:
        print(f"  Missing files:    {result['missing']}")
    if result["extra"]:
        print(f"  Extra files:      {result['extra']}")
    if result["mismatched"]:
        print(f"  Modified files:")
        for m in result["mismatched"]:
            print(f"    {m['name']}: expected {m['expected'][:10]} → "
                  f"actual {m['actual'][:10]}")
    sys.exit(0 if result["ok"] else 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Axion AI — data version CLI")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("snapshot", help="Take a snapshot of data/")
    ps.add_argument("--message", "-m", default="", help="Snapshot message")
    ps.set_defaults(func=cmd_snapshot)

    pl = sub.add_parser("list", help="List existing snapshots")
    pl.set_defaults(func=cmd_list)

    psh = sub.add_parser("show", help="Show one snapshot's manifest")
    psh.add_argument("id")
    psh.set_defaults(func=cmd_show)

    pd = sub.add_parser("diff", help="Diff two snapshots")
    pd.add_argument("a"); pd.add_argument("b")
    pd.set_defaults(func=cmd_diff)

    pv = sub.add_parser("verify",
                         help="Verify the current data/ matches a snapshot")
    pv.add_argument("id")
    pv.set_defaults(func=cmd_verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
