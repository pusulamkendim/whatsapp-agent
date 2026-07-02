"""add llm price sync metadata

Revision ID: 20260702_0003
Revises: 20260702_0002
Create Date: 2026-07-02
"""
from alembic import op

revision = "20260702_0003"
down_revision = "20260702_0002"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for statement in [
            "ALTER TABLE llm_models ADD COLUMN IF NOT EXISTS pricing_source VARCHAR DEFAULT ''",
            "ALTER TABLE llm_models ADD COLUMN IF NOT EXISTS pricing_checked_at TIMESTAMP",
            "ALTER TABLE llm_models ADD COLUMN IF NOT EXISTS pricing_sync_error TEXT DEFAULT ''",
        ]:
            op.execute(statement)
        return

    existing_columns = {
        row[1] for row in bind.exec_driver_sql("PRAGMA table_info(llm_models)").fetchall()
    }
    additions = {
        "pricing_source": "VARCHAR DEFAULT ''",
        "pricing_checked_at": "DATETIME",
        "pricing_sync_error": "TEXT DEFAULT ''",
    }
    for column, ddl in additions.items():
        if column not in existing_columns:
            op.execute(f"ALTER TABLE llm_models ADD COLUMN {column} {ddl}")


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for column in ["pricing_sync_error", "pricing_checked_at", "pricing_source"]:
            op.execute(f"ALTER TABLE llm_models DROP COLUMN IF EXISTS {column}")
