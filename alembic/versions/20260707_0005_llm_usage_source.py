"""add llm usage source metadata

Revision ID: 20260707_0005
Revises: 20260707_0004
Create Date: 2026-07-07
"""
from alembic import op

revision = "20260707_0005"
down_revision = "20260707_0004"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for statement in [
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT ''",
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS operation VARCHAR DEFAULT ''",
            "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_source ON llm_usage_logs (source)",
            "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_operation ON llm_usage_logs (operation)",
        ]:
            op.execute(statement)
        return

    existing_columns = {
        row[1] for row in bind.exec_driver_sql("PRAGMA table_info(llm_usage_logs)").fetchall()
    }
    additions = {
        "source": "VARCHAR DEFAULT ''",
        "operation": "VARCHAR DEFAULT ''",
    }
    for column, ddl in additions.items():
        if column not in existing_columns:
            op.execute(f"ALTER TABLE llm_usage_logs ADD COLUMN {column} {ddl}")
    op.execute("CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_source ON llm_usage_logs (source)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_operation ON llm_usage_logs (operation)")


def downgrade():
    bind = op.get_bind()
    for index_name in [
        "ix_llm_usage_logs_source",
        "ix_llm_usage_logs_operation",
    ]:
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
    if bind.dialect.name == "postgresql":
        for column in ["operation", "source"]:
            op.execute(f"ALTER TABLE llm_usage_logs DROP COLUMN IF EXISTS {column}")
