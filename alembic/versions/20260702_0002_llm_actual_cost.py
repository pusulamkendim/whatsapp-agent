"""add actual llm cost metadata

Revision ID: 20260702_0002
Revises: 20260701_0001
Create Date: 2026-07-02
"""
from alembic import op

revision = "20260702_0002"
down_revision = "20260701_0001"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for statement in [
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS actual_cost DOUBLE PRECISION",
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS generation_id VARCHAR DEFAULT ''",
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS actual_model_ref VARCHAR DEFAULT ''",
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS actual_provider VARCHAR DEFAULT ''",
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS router VARCHAR DEFAULT ''",
            "ALTER TABLE llm_usage_logs ADD COLUMN IF NOT EXISTS cost_details_json TEXT DEFAULT ''",
            "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_generation_id ON llm_usage_logs (generation_id)",
            "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_actual_model_ref ON llm_usage_logs (actual_model_ref)",
        ]:
            op.execute(statement)
        return

    existing_columns = {
        row[1] for row in bind.exec_driver_sql("PRAGMA table_info(llm_usage_logs)").fetchall()
    }
    additions = {
        "actual_cost": "FLOAT",
        "generation_id": "VARCHAR DEFAULT ''",
        "actual_model_ref": "VARCHAR DEFAULT ''",
        "actual_provider": "VARCHAR DEFAULT ''",
        "router": "VARCHAR DEFAULT ''",
        "cost_details_json": "TEXT DEFAULT ''",
    }
    for column, ddl in additions.items():
        if column not in existing_columns:
            op.execute(f"ALTER TABLE llm_usage_logs ADD COLUMN {column} {ddl}")
    op.execute("CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_generation_id ON llm_usage_logs (generation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_actual_model_ref ON llm_usage_logs (actual_model_ref)")


def downgrade():
    bind = op.get_bind()
    for index_name in [
        "ix_llm_usage_logs_generation_id",
        "ix_llm_usage_logs_actual_model_ref",
    ]:
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
    if bind.dialect.name == "postgresql":
        for column in [
            "cost_details_json",
            "router",
            "actual_provider",
            "actual_model_ref",
            "generation_id",
            "actual_cost",
        ]:
            op.execute(f"ALTER TABLE llm_usage_logs DROP COLUMN IF EXISTS {column}")
