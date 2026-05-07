"""
Axion AI — Users repository
=============================

CRUD over the `users` table (created by migration 002). Built on top of
the existing `DbClient` connection so we don't introduce a second
connection pool.

The repository is deliberately tiny — Axion's user count is in the dozens
at most for a pilot, so SELECT-by-email + INSERT is enough. No ORM, no
migrations from code, no async — psycopg2 + raw SQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class UserRecord:
    id:            int
    email:         str
    password_hash: str
    role:          str
    active:        bool
    created_at:    Optional[datetime] = None

    def to_public_dict(self) -> dict:
        """Serialization safe to return over the API (no password hash)."""
        return {
            "id":         self.id,
            "email":      self.email,
            "role":       self.role,
            "active":     self.active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


VALID_ROLES = ("viewer", "operator", "manager")


class UserRepository:
    """Lightweight CRUD wrapper around an existing DbClient."""

    def __init__(self, db_client) -> None:
        # Late import to avoid a hard dependency on psycopg2 at module load
        # (the rest of the app already imports it via db.client).
        self.db = db_client

    # ---- queries ----

    def get_by_email(self, email: str) -> Optional[UserRecord]:
        if not email:
            return None
        with self.db._conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, password_hash, role, active, created_at "
                "FROM users WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
        return _row_to_user(row)

    def get_by_id(self, user_id: int) -> Optional[UserRecord]:
        with self.db._conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, password_hash, role, active, created_at "
                "FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
        return _row_to_user(row)

    def list_all(self) -> List[UserRecord]:
        with self.db._conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, password_hash, role, active, created_at "
                "FROM users ORDER BY id"
            )
            rows = cur.fetchall()
        return [_row_to_user(r) for r in rows if r is not None]

    # ---- mutations ----

    def create(
        self, *, email: str, password_hash: str, role: str,
        active: bool = True,
    ) -> UserRecord:
        _validate_role(role)
        with self.db._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, password_hash, role, active) "
                "VALUES (%s, %s, %s, %s) "
                "RETURNING id, email, password_hash, role, active, created_at",
                (email, password_hash, role, active),
            )
            row = cur.fetchone()
        self.db._conn.commit()
        return _row_to_user(row)

    def update_role(self, user_id: int, role: str) -> None:
        _validate_role(role)
        with self.db._conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET role = %s WHERE id = %s",
                (role, user_id),
            )
        self.db._conn.commit()

    def update_password(self, user_id: int, password_hash: str) -> None:
        with self.db._conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (password_hash, user_id),
            )
        self.db._conn.commit()

    def set_active(self, user_id: int, active: bool) -> None:
        with self.db._conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET active = %s WHERE id = %s",
                (active, user_id),
            )
        self.db._conn.commit()

    def delete(self, user_id: int) -> None:
        with self.db._conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        self.db._conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_user(row) -> Optional[UserRecord]:
    if row is None:
        return None
    return UserRecord(
        id=row[0], email=row[1], password_hash=row[2],
        role=row[3], active=bool(row[4]),
        created_at=row[5] if len(row) > 5 else None,
    )


def _validate_role(role: str) -> None:
    if role not in VALID_ROLES:
        raise ValueError(
            f"role must be one of {VALID_ROLES}, got {role!r}"
        )
