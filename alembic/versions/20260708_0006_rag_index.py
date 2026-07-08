"""add rag index tables

Revision ID: 20260708_0006
Revises: 20260707_0005
Create Date: 2026-07-08
"""
from alembic import op
import sqlalchemy as sa

revision = "20260708_0006"
down_revision = "20260707_0005"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "knowledge_chunks" not in tables:
        op.create_table(
            "knowledge_chunks",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("knowledge_base_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id"), nullable=False),
            sa.Column("document_id", sa.Integer(), sa.ForeignKey("knowledge_documents.id"), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("title_path", sa.Text(), server_default=""),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("content_hash", sa.String(), nullable=False),
            sa.Column("token_count", sa.Integer(), server_default="0"),
            sa.Column("metadata_json", sa.Text(), server_default="{}"),
            sa.Column("active", sa.Boolean(), server_default=sa.true()),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )
    _create_index_if_missing(inspector, "knowledge_chunks", "ix_knowledge_chunks_knowledge_base_id", ["knowledge_base_id"])
    _create_index_if_missing(inspector, "knowledge_chunks", "ix_knowledge_chunks_document_id", ["document_id"])
    _create_index_if_missing(inspector, "knowledge_chunks", "ix_knowledge_chunks_content_hash", ["content_hash"])
    _create_index_if_missing(inspector, "knowledge_chunks", "ix_knowledge_chunks_active", ["active"])

    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "knowledge_embeddings" not in tables:
        op.create_table(
            "knowledge_embeddings",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("chunk_id", sa.Integer(), sa.ForeignKey("knowledge_chunks.id"), nullable=False),
            sa.Column("embedding_model", sa.String(), nullable=False),
            sa.Column("embedding_dim", sa.Integer(), nullable=False),
            sa.Column("vector_json", sa.Text(), nullable=False),
            sa.Column("content_hash", sa.String(), nullable=False),
            sa.Column("status", sa.String(), server_default="ready"),
            sa.Column("error_message", sa.Text(), server_default=""),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )
    _create_index_if_missing(inspector, "knowledge_embeddings", "ix_knowledge_embeddings_chunk_id", ["chunk_id"])
    _create_index_if_missing(inspector, "knowledge_embeddings", "ix_knowledge_embeddings_embedding_model", ["embedding_model"])
    _create_index_if_missing(inspector, "knowledge_embeddings", "ix_knowledge_embeddings_content_hash", ["content_hash"])
    _create_index_if_missing(inspector, "knowledge_embeddings", "ix_knowledge_embeddings_status", ["status"])

    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "rag_query_logs" not in tables:
        op.create_table(
            "rag_query_logs",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("agent_id", sa.Integer(), sa.ForeignKey("agents.id"), nullable=True),
            sa.Column("external_user_id", sa.String(), server_default=""),
            sa.Column("query", sa.Text(), nullable=False),
            sa.Column("rewritten_query", sa.Text(), server_default=""),
            sa.Column("retrieved_chunk_ids_json", sa.Text(), server_default="[]"),
            sa.Column("scores_json", sa.Text(), server_default="[]"),
            sa.Column("selected_context_chars", sa.Integer(), server_default="0"),
            sa.Column("retrieval_latency_ms", sa.Integer(), nullable=True),
            sa.Column("rerank_latency_ms", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime()),
        )
    _create_index_if_missing(inspector, "rag_query_logs", "ix_rag_query_logs_agent_id", ["agent_id"])
    _create_index_if_missing(inspector, "rag_query_logs", "ix_rag_query_logs_external_user_id", ["external_user_id"])
    _create_index_if_missing(inspector, "rag_query_logs", "ix_rag_query_logs_created_at", ["created_at"])


def downgrade():
    op.drop_index("ix_rag_query_logs_created_at", table_name="rag_query_logs")
    op.drop_index("ix_rag_query_logs_external_user_id", table_name="rag_query_logs")
    op.drop_index("ix_rag_query_logs_agent_id", table_name="rag_query_logs")
    op.drop_table("rag_query_logs")

    op.drop_index("ix_knowledge_embeddings_status", table_name="knowledge_embeddings")
    op.drop_index("ix_knowledge_embeddings_content_hash", table_name="knowledge_embeddings")
    op.drop_index("ix_knowledge_embeddings_embedding_model", table_name="knowledge_embeddings")
    op.drop_index("ix_knowledge_embeddings_chunk_id", table_name="knowledge_embeddings")
    op.drop_table("knowledge_embeddings")

    op.drop_index("ix_knowledge_chunks_active", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_content_hash", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_document_id", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_knowledge_base_id", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")


def _create_index_if_missing(inspector, table_name: str, index_name: str, columns: list[str]):
    existing = {index["name"] for index in inspector.get_indexes(table_name)}
    if index_name not in existing:
        op.create_index(index_name, table_name, columns)
