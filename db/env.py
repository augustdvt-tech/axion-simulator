"""Alembic env.py — reads AXION_DB_URL from the environment."""

from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Alembic config object ──────────────────────────────────────────────────
config = context.config

if config.config_file_name:
    fileConfig(config.config_file_name)

# ── Connection URL ─────────────────────────────────────────────────────────
# Prefer AXION_DB_URL env var; fall back to alembic.ini sqlalchemy.url.
db_url = os.environ.get("AXION_DB_URL") or config.get_main_option("sqlalchemy.url")
if not db_url:
    raise RuntimeError(
        "No database URL found. Set the AXION_DB_URL environment variable "
        "or sqlalchemy.url in db/alembic.ini."
    )
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = None


# ── Offline mode ───────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode ────────────────────────────────────────────────────────────

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
