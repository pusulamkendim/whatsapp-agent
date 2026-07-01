"""add llm admin tables

Revision ID: 20260701_0001
Revises:
Create Date: 2026-07-01
"""
from alembic import op

revision = "20260701_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        _upgrade_postgres()
    else:
        _upgrade_sqlite()


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TABLE IF EXISTS llm_usage_logs")
        op.execute("DROP TABLE IF EXISTS llm_models")
        op.execute("DROP TABLE IF EXISTS llm_providers")
    else:
        op.execute("DROP TABLE IF EXISTS llm_usage_logs")
        op.execute("DROP TABLE IF EXISTS llm_models")
        op.execute("DROP TABLE IF EXISTS llm_providers")


def _upgrade_postgres():
    for statement in [
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS fallback_model VARCHAR DEFAULT ''",
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS temperature DOUBLE PRECISION DEFAULT 0.7",
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS max_tokens INTEGER",
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS timeout_seconds INTEGER DEFAULT 60",
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS daily_budget_limit DOUBLE PRECISION",
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS monthly_budget_limit DOUBLE PRECISION",
        "ALTER TABLE agents ADD COLUMN IF NOT EXISTS failover_enabled BOOLEAN DEFAULT TRUE",
        """
        CREATE TABLE IF NOT EXISTS llm_providers (
            id SERIAL PRIMARY KEY,
            slug VARCHAR NOT NULL UNIQUE,
            name VARCHAR NOT NULL,
            base_url VARCHAR DEFAULT '',
            api_key_env VARCHAR DEFAULT '',
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_llm_providers_slug ON llm_providers (slug)",
        """
        CREATE TABLE IF NOT EXISTS llm_models (
            id SERIAL PRIMARY KEY,
            provider_id INTEGER NOT NULL REFERENCES llm_providers(id),
            slug VARCHAR NOT NULL,
            display_name VARCHAR NOT NULL,
            model_ref VARCHAR NOT NULL UNIQUE,
            supports_tools BOOLEAN DEFAULT FALSE,
            supports_vision BOOLEAN DEFAULT FALSE,
            context_window INTEGER,
            input_price DOUBLE PRECISION,
            output_price DOUBLE PRECISION,
            rate_limit_rpm INTEGER,
            rate_limit_tpm INTEGER,
            active BOOLEAN DEFAULT TRUE,
            is_default BOOLEAN DEFAULT FALSE,
            notes TEXT DEFAULT '',
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_llm_models_provider_id ON llm_models (provider_id)",
        "CREATE INDEX IF NOT EXISTS ix_llm_models_slug ON llm_models (slug)",
        "CREATE INDEX IF NOT EXISTS ix_llm_models_model_ref ON llm_models (model_ref)",
        """
        CREATE TABLE IF NOT EXISTS llm_usage_logs (
            id SERIAL PRIMARY KEY,
            agent_id INTEGER REFERENCES agents(id),
            provider VARCHAR DEFAULT '',
            model_ref VARCHAR DEFAULT '',
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            estimated_cost DOUBLE PRECISION,
            latency_ms INTEGER,
            success BOOLEAN DEFAULT TRUE,
            error_code VARCHAR DEFAULT '',
            error_message TEXT DEFAULT '',
            created_at TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_agent_id ON llm_usage_logs (agent_id)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_provider ON llm_usage_logs (provider)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_model_ref ON llm_usage_logs (model_ref)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_success ON llm_usage_logs (success)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_created_at ON llm_usage_logs (created_at)",
    ]:
        op.execute(statement)


def _upgrade_sqlite():
    existing_agent_columns = {
        row[1] for row in op.get_bind().exec_driver_sql("PRAGMA table_info(agents)").fetchall()
    }
    additions = {
        "fallback_model": "VARCHAR DEFAULT ''",
        "temperature": "FLOAT DEFAULT 0.7",
        "max_tokens": "INTEGER",
        "timeout_seconds": "INTEGER DEFAULT 60",
        "daily_budget_limit": "FLOAT",
        "monthly_budget_limit": "FLOAT",
        "failover_enabled": "BOOLEAN DEFAULT 1",
    }
    for column, ddl in additions.items():
        if column not in existing_agent_columns:
            op.execute(f"ALTER TABLE agents ADD COLUMN {column} {ddl}")

    for statement in [
        """
        CREATE TABLE IF NOT EXISTS llm_providers (
            id INTEGER NOT NULL PRIMARY KEY,
            slug VARCHAR NOT NULL UNIQUE,
            name VARCHAR NOT NULL,
            base_url VARCHAR DEFAULT '',
            api_key_env VARCHAR DEFAULT '',
            active BOOLEAN DEFAULT 1,
            created_at DATETIME,
            updated_at DATETIME
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_llm_providers_slug ON llm_providers (slug)",
        """
        CREATE TABLE IF NOT EXISTS llm_models (
            id INTEGER NOT NULL PRIMARY KEY,
            provider_id INTEGER NOT NULL REFERENCES llm_providers(id),
            slug VARCHAR NOT NULL,
            display_name VARCHAR NOT NULL,
            model_ref VARCHAR NOT NULL UNIQUE,
            supports_tools BOOLEAN DEFAULT 0,
            supports_vision BOOLEAN DEFAULT 0,
            context_window INTEGER,
            input_price FLOAT,
            output_price FLOAT,
            rate_limit_rpm INTEGER,
            rate_limit_tpm INTEGER,
            active BOOLEAN DEFAULT 1,
            is_default BOOLEAN DEFAULT 0,
            notes TEXT DEFAULT '',
            created_at DATETIME,
            updated_at DATETIME
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_llm_models_provider_id ON llm_models (provider_id)",
        "CREATE INDEX IF NOT EXISTS ix_llm_models_slug ON llm_models (slug)",
        "CREATE INDEX IF NOT EXISTS ix_llm_models_model_ref ON llm_models (model_ref)",
        """
        CREATE TABLE IF NOT EXISTS llm_usage_logs (
            id INTEGER NOT NULL PRIMARY KEY,
            agent_id INTEGER REFERENCES agents(id),
            provider VARCHAR DEFAULT '',
            model_ref VARCHAR DEFAULT '',
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            estimated_cost FLOAT,
            latency_ms INTEGER,
            success BOOLEAN DEFAULT 1,
            error_code VARCHAR DEFAULT '',
            error_message TEXT DEFAULT '',
            created_at DATETIME
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_agent_id ON llm_usage_logs (agent_id)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_provider ON llm_usage_logs (provider)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_model_ref ON llm_usage_logs (model_ref)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_success ON llm_usage_logs (success)",
        "CREATE INDEX IF NOT EXISTS ix_llm_usage_logs_created_at ON llm_usage_logs (created_at)",
    ]:
        op.execute(statement)
