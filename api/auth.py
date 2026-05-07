"""
Axion AI — JWT + password hashing utilities
============================================

Pure helpers for the auth layer. No FastAPI imports, no DB imports, no
global state — every function is independently testable.

Token shape
-----------
Both access and refresh tokens are HS256-signed JWTs with the following
claims:

    sub   email of the authenticated user
    uid   numeric user id (matches users.id in the DB)
    role  one of: viewer | operator | manager
    type  "access" | "refresh"
    iat   issued-at unix timestamp
    exp   expiration unix timestamp

Access tokens are short-lived (default 30 min) and used on every API
request. Refresh tokens are long-lived (default 7 days) and only accepted
by `/api/auth/refresh`. Distinct `type` claims keep the two from being
interchangeable.

Configuration via environment:
    AXION_JWT_SECRET           — HS256 secret. Required to enable JWT auth.
    AXION_JWT_ACCESS_MINUTES   — access token TTL (default: 30)
    AXION_JWT_REFRESH_DAYS     — refresh token TTL (default: 7)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import bcrypt
import jwt


JWT_ALGORITHM = "HS256"


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class AuthError(Exception):
    """Base for auth-layer errors. Caller should map to HTTP 401."""


class InvalidTokenError(AuthError):
    """JWT was malformed, signature didn't verify, or required claim missing."""


class TokenExpiredError(AuthError):
    """JWT 'exp' claim has passed."""


class TokenTypeMismatchError(AuthError):
    """Used an access token where a refresh was expected, or vice versa."""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def jwt_secret() -> Optional[str]:
    """Return the configured HS256 secret, or None if JWT auth is disabled."""
    s = os.environ.get("AXION_JWT_SECRET", "").strip()
    return s or None


def access_ttl_seconds() -> int:
    return int(os.environ.get("AXION_JWT_ACCESS_MINUTES", "30")) * 60


def refresh_ttl_seconds() -> int:
    return int(os.environ.get("AXION_JWT_REFRESH_DAYS", "7")) * 24 * 3600


# ─────────────────────────────────────────────────────────────────────────────
# Password hashing (bcrypt)
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    if not plain:
        raise ValueError("password must not be empty")
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Token encode / decode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TokenClaims:
    sub:  str
    uid:  int
    role: str
    type: str
    iat:  int
    exp:  int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub": self.sub, "uid": self.uid, "role": self.role,
            "type": self.type, "iat": self.iat, "exp": self.exp,
        }


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def encode_token(
    *,
    secret: str,
    sub: str,
    uid: int,
    role: str,
    token_type: str,
    ttl_seconds: int,
    issued_at: Optional[int] = None,
) -> str:
    """Build and sign a JWT with the given claims."""
    if token_type not in ("access", "refresh"):
        raise ValueError(f"token_type must be 'access' or 'refresh', got {token_type!r}")
    iat = issued_at if issued_at is not None else _now_ts()
    payload = {
        "sub":  sub,
        "uid":  uid,
        "role": role,
        "type": token_type,
        "iat":  iat,
        "exp":  iat + int(ttl_seconds),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_token(token: str, secret: str,
                  expected_type: Optional[str] = None) -> TokenClaims:
    """Verify signature + expiration. Optionally enforce token type.

    Raises:
        TokenExpiredError       if the JWT has expired
        TokenTypeMismatchError  if expected_type doesn't match the claim
        InvalidTokenError       for any other validation failure
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredError(str(e)) from e
    except jwt.PyJWTError as e:
        raise InvalidTokenError(str(e)) from e

    for required in ("sub", "uid", "role", "type", "iat", "exp"):
        if required not in payload:
            raise InvalidTokenError(f"Missing claim: {required}")

    if expected_type is not None and payload["type"] != expected_type:
        raise TokenTypeMismatchError(
            f"Expected token type {expected_type!r}, got {payload['type']!r}"
        )

    return TokenClaims(
        sub=str(payload["sub"]),
        uid=int(payload["uid"]),
        role=str(payload["role"]),
        type=str(payload["type"]),
        iat=int(payload["iat"]),
        exp=int(payload["exp"]),
    )


def issue_token_pair(
    *,
    secret: str,
    sub: str,
    uid: int,
    role: str,
    access_ttl: Optional[int] = None,
    refresh_ttl: Optional[int] = None,
) -> Tuple[str, str, int]:
    """Build a fresh (access, refresh, access_expires_in_seconds) tuple."""
    a_ttl = access_ttl  if access_ttl  is not None else access_ttl_seconds()
    r_ttl = refresh_ttl if refresh_ttl is not None else refresh_ttl_seconds()
    iat = _now_ts()
    access  = encode_token(secret=secret, sub=sub, uid=uid, role=role,
                            token_type="access",  ttl_seconds=a_ttl,
                            issued_at=iat)
    refresh = encode_token(secret=secret, sub=sub, uid=uid, role=role,
                            token_type="refresh", ttl_seconds=r_ttl,
                            issued_at=iat)
    return access, refresh, a_ttl


def extract_bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Pull the JWT out of an `Authorization: Bearer <token>` header."""
    if not header_value:
        return None
    parts = header_value.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]
