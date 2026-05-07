"""
Axion AI — User management CLI
================================

Create, list, update, and deactivate users in the `users` table. Use this
to seed the first admin / manager during deployment.

Examples:
    python scripts/users.py create --email a@b.com --role manager
    python scripts/users.py list
    python scripts/users.py set-role --email a@b.com --role operator
    python scripts/users.py reset-password --email a@b.com
    python scripts/users.py deactivate --email a@b.com

Reads the same AXION_DB_URL the API server uses.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.auth import hash_password
from db.client import DbClient
from db.users import UserRepository, VALID_ROLES


def _connect() -> DbClient:
    url = os.environ.get("AXION_DB_URL")
    if not url:
        print("ERROR: AXION_DB_URL is not set.", file=sys.stderr)
        sys.exit(1)
    db = DbClient(url)
    db.connect()
    return db


def _read_password(prompt: str = "Password: ") -> str:
    pw = getpass.getpass(prompt)
    if not pw:
        print("ERROR: password must not be empty.", file=sys.stderr)
        sys.exit(2)
    confirm = getpass.getpass("Confirm:  ")
    if pw != confirm:
        print("ERROR: passwords do not match.", file=sys.stderr)
        sys.exit(2)
    return pw


def cmd_create(args) -> None:
    db   = _connect()
    repo = UserRepository(db)
    if repo.get_by_email(args.email):
        print(f"User {args.email} already exists.", file=sys.stderr)
        sys.exit(3)
    pw = _read_password()
    user = repo.create(
        email=args.email,
        password_hash=hash_password(pw),
        role=args.role,
        active=True,
    )
    print(f"  Created user id={user.id} email={user.email} role={user.role}")


def cmd_list(args) -> None:
    db   = _connect()
    repo = UserRepository(db)
    users = repo.list_all()
    if not users:
        print("(no users)")
        return
    print(f"{'ID':>4}  {'EMAIL':30s}  {'ROLE':10s}  ACTIVE")
    print("-" * 60)
    for u in users:
        print(f"{u.id:>4}  {u.email:30s}  {u.role:10s}  "
              f"{'yes' if u.active else 'no'}")


def cmd_set_role(args) -> None:
    db   = _connect()
    repo = UserRepository(db)
    user = repo.get_by_email(args.email)
    if user is None:
        print(f"User {args.email} not found.", file=sys.stderr)
        sys.exit(4)
    repo.update_role(user.id, args.role)
    print(f"  Updated role of {args.email} → {args.role}")


def cmd_reset_password(args) -> None:
    db   = _connect()
    repo = UserRepository(db)
    user = repo.get_by_email(args.email)
    if user is None:
        print(f"User {args.email} not found.", file=sys.stderr)
        sys.exit(4)
    pw = _read_password("New password: ")
    repo.update_password(user.id, hash_password(pw))
    print(f"  Password updated for {args.email}")


def cmd_deactivate(args) -> None:
    db   = _connect()
    repo = UserRepository(db)
    user = repo.get_by_email(args.email)
    if user is None:
        print(f"User {args.email} not found.", file=sys.stderr)
        sys.exit(4)
    repo.set_active(user.id, False)
    print(f"  Deactivated {args.email}")


def main() -> None:
    p = argparse.ArgumentParser(description="Axion AI — user management")
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("create", help="Create a new user")
    pc.add_argument("--email", required=True)
    pc.add_argument("--role", required=True, choices=VALID_ROLES)
    pc.set_defaults(func=cmd_create)

    pl = sub.add_parser("list", help="List all users")
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("set-role", help="Change a user's role")
    pr.add_argument("--email", required=True)
    pr.add_argument("--role", required=True, choices=VALID_ROLES)
    pr.set_defaults(func=cmd_set_role)

    pp = sub.add_parser("reset-password", help="Reset a user's password")
    pp.add_argument("--email", required=True)
    pp.set_defaults(func=cmd_reset_password)

    pd = sub.add_parser("deactivate", help="Deactivate a user")
    pd.add_argument("--email", required=True)
    pd.set_defaults(func=cmd_deactivate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
