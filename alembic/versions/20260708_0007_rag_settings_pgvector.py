"""add rag settings and optional pgvector column

Revision ID: 20260708_0007
Revises: 20260708_0006
Create Date: 2026-07-08
"""
from alembic import op
import sqlalchemy as sa

revision = "20260708_0007"
down_revision = "20260708_0006"
branch_labels = None
depends_on = None


AGENT_COLUMNS = {
    "rag_top_k": "INTEGER DEFAULT 20",
    "rag_final_chunks": "INTEGER DEFAULT 6",
    "rag_min_score": "FLOAT",
    "rag_max_context_chars": "INTEGER DEFAULT 12000",
    "rag_hybrid_search": "BOOLEAN DEFAULT TRUE",
    "rag_rerank_enabled": "BOOLEAN DEFAULT FALSE",
    "rag_query_rewrite_enabled": "BOOLEAN DEFAULT FALSE",
    "rag_embedding_model": "VARCHAR DEFAULT ''",
}

RAG_LOG_COLUMNS = {
    "answer_latency_ms": "INTEGER",
    "model_ref": "VARCHAR DEFAULT ''",
    "prompt_tokens": "INTEGER",
    "completion_tokens": "INTEGER",
    "total_tokens": "INTEGER",
}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    agent_columns = {column["name"] for column in inspector.get_columns("agents")}

    for column, ddl in AGENT_COLUMNS.items():
        if column not in agent_columns:
            op.execute(f"ALTER TABLE agents ADD COLUMN {column} {ddl}")

    rag_log_columns = {column["name"] for column in inspector.get_columns("rag_query_logs")}
    for column, ddl in RAG_LOG_COLUMNS.items():
        if column not in rag_log_columns:
            op.execute(f"ALTER TABLE rag_query_logs ADD COLUMN {column} {ddl}")

    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        embedding_columns = {column["name"] for column in inspector.get_columns("knowledge_embeddings")}
        if "embedding_vector" not in embedding_columns:
            op.execute("ALTER TABLE knowledge_embeddings ADD COLUMN embedding_vector vector")


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE knowledge_embeddings DROP COLUMN IF EXISTS embedding_vector")
    for column in reversed(list(AGENT_COLUMNS.keys())):
        if bind.dialect.name == "postgresql":
            op.execute(f"ALTER TABLE agents DROP COLUMN IF EXISTS {column}")
    if bind.dialect.name == "postgresql":
        for column in reversed(list(RAG_LOG_COLUMNS.keys())):
            op.execute(f"ALTER TABLE rag_query_logs DROP COLUMN IF EXISTS {column}")
