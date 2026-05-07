"""Users table for JWT-based RBAC.

Revision ID: 002
Revises: 001
"""

from alembic import op
import sqlalchemy as sa


revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id",            sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("email",         sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role",          sa.String(32),  nullable=False),
        sa.Column("active",        sa.Boolean,     nullable=False,
                   server_default=sa.text("TRUE")),
        sa.Column("created_at",    sa.TIMESTAMP(timezone=True),
                   nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "role IN ('viewer','operator','manager')", name="users_role_chk",
        ),
    )
    op.create_index("idx_users_email", "users", ["email"])


def downgrade() -> None:
    op.drop_index("idx_users_email", table_name="users")
    op.drop_table("users")
