"""Initial schema: process_samples hypertable + scenarios, recommendations, decisions.

Revision ID: 001
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── TimescaleDB extension ──────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # ── scenarios ─────────────────────────────────────────────────────────
    op.create_table(
        "scenarios",
        sa.Column("name",        sa.Text, primary_key=True),
        sa.Column("duration_h",  sa.Float,   nullable=True),
        sa.Column("n_samples",   sa.Integer, nullable=True),
        sa.Column("ingested_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("notes",       sa.Text,    nullable=True),
    )

    # ── process_samples ────────────────────────────────────────────────────
    # Wide table: one row per (timestamp, scenario). Sensor tags become columns.
    # Dots in column names are quoted; PostgreSQL handles them fine.
    op.create_table(
        "process_samples",
        sa.Column("timestamp",          sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("scenario",           sa.Text,    nullable=False),
        # CSTR
        sa.Column("cstr_T_R_C",         sa.Float,   nullable=True),
        sa.Column("cstr_T_J_C",         sa.Float,   nullable=True),
        sa.Column("cstr_C_A",           sa.Float,   nullable=True),
        sa.Column("cstr_F_feed",        sa.Float,   nullable=True),
        sa.Column("cstr_F_cool",        sa.Float,   nullable=True),
        sa.Column("cstr_T_feed_C",      sa.Float,   nullable=True),
        sa.Column("cstr_T_cool_in_C",   sa.Float,   nullable=True),
        sa.Column("cstr_P_R",           sa.Float,   nullable=True),
        sa.Column("cstr_conversion",    sa.Float,   nullable=True),
        # Distillation column
        sa.Column("column_x_D",         sa.Float,   nullable=True),
        sa.Column("column_x_B_A",       sa.Float,   nullable=True),
        sa.Column("column_purity_B",    sa.Float,   nullable=True),
        sa.Column("column_T_top_C",     sa.Float,   nullable=True),
        sa.Column("column_T_bot_C",     sa.Float,   nullable=True),
        sa.Column("column_RR",          sa.Float,   nullable=True),
        sa.Column("column_F_vap_kgh",   sa.Float,   nullable=True),
        sa.Column("column_Q_reb_kW",    sa.Float,   nullable=True),
        sa.Column("column_P_top_bar",   sa.Float,   nullable=True),
        sa.Column("column_P_bot_bar",   sa.Float,   nullable=True),
    )

    # Unique constraint so re-ingest is idempotent via ON CONFLICT DO NOTHING
    op.create_index(
        "uq_process_samples_ts_scenario",
        "process_samples",
        ["timestamp", "scenario"],
        unique=True,
    )

    # Promote to TimescaleDB hypertable, partitioned by week
    op.execute(
        "SELECT create_hypertable('process_samples', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 week', if_not_exists => TRUE)"
    )

    # ── recommendations ────────────────────────────────────────────────────
    op.create_table(
        "recommendations",
        sa.Column("id",          sa.Text,    primary_key=True),
        sa.Column("scenario",    sa.Text,    nullable=False),
        sa.Column("timestamp",   sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("rule_id",     sa.Text,    nullable=False),
        sa.Column("urgency",     sa.Text,    nullable=False),
        sa.Column("confidence",  sa.Float,   nullable=True),
        sa.Column("diagnosis",   sa.Text,    nullable=True),
        sa.Column("action",      sa.Text,    nullable=True),
        sa.Column("status",      sa.Text,    nullable=True, server_default="pending"),
        sa.Column("created_at",  sa.TIMESTAMP(timezone=True),
                  server_default=sa.text("NOW()")),
    )
    op.create_index("ix_recommendations_scenario_ts",
                    "recommendations", ["scenario", "timestamp"])

    # ── decisions ─────────────────────────────────────────────────────────
    op.create_table(
        "decisions",
        sa.Column("id",              sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("recommendation_id", sa.Text,  nullable=False),
        sa.Column("decision",        sa.Text,    nullable=False),   # accept/modify/reject
        sa.Column("rationale",       sa.Text,    nullable=True),
        sa.Column("decided_at",      sa.TIMESTAMP(timezone=True),
                  server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["recommendation_id"], ["recommendations.id"],
                                ondelete="CASCADE"),
    )
    op.create_index("ix_decisions_recommendation_id",
                    "decisions", ["recommendation_id"])


def downgrade() -> None:
    op.drop_table("decisions")
    op.drop_table("recommendations")
    op.drop_table("process_samples")
    op.drop_table("scenarios")
    op.execute("DROP EXTENSION IF EXISTS timescaledb")
