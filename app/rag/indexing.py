from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.llm import parse_model_ref
from app.llm_usage import add_llm_usage_log
from app.models import KnowledgeBase, KnowledgeChunk, KnowledgeDocument, KnowledgeEmbedding
from app.rag.chunking import ChunkSpec, chunk_document
from app.rag.embeddings import embed_texts, embedding_model


def index_document(document: KnowledgeDocument, db: Session, model_ref: str | None = None, agent_id: int | None = None) -> dict:
    model_ref = model_ref or embedding_model()
    chunks = chunk_document(document.content or "", filename=document.filename or "")
    if _document_index_is_current(document, chunks, model_ref):
        return {
            "ok": True,
            "document_id": document.id,
            "status": "unchanged",
            "chunk_count": len(chunks),
        }

    db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == document.id,
        KnowledgeChunk.active == True,
    ).update({"active": False, "updated_at": datetime.now(timezone.utc)})
    db.flush()

    if not chunks:
        return {"ok": True, "document_id": document.id, "status": "empty", "chunk_count": 0}

    created_chunks = [_create_chunk(document, spec, db) for spec in chunks]
    db.flush()

    started = time.perf_counter()
    try:
        vectors = embed_texts([chunk.content for chunk in created_chunks], model_ref=model_ref)
        _log_embedding_usage(
            db,
            model_ref,
            created_chunks,
            success=True,
            latency_ms=_elapsed_ms(started),
            agent_id=agent_id,
            operation="embedding:index",
        )
        for chunk, vector in zip(created_chunks, vectors):
            embedding = KnowledgeEmbedding(
                chunk_id=chunk.id,
                embedding_model=model_ref,
                embedding_dim=len(vector),
                vector_json=json.dumps(vector, separators=(",", ":")),
                content_hash=chunk.content_hash,
                status="ready",
                error_message="",
            )
            db.add(embedding)
            db.flush()
            _store_pgvector(embedding.id, vector, db)
        status = "indexed"
        ok = True
        error = ""
    except Exception as exc:
        _log_embedding_usage(
            db,
            model_ref,
            created_chunks,
            success=False,
            latency_ms=_elapsed_ms(started),
            error=exc,
            agent_id=agent_id,
            operation="embedding:index",
        )
        for chunk in created_chunks:
            db.add(KnowledgeEmbedding(
                chunk_id=chunk.id,
                embedding_model=model_ref,
                embedding_dim=0,
                vector_json="[]",
                content_hash=chunk.content_hash,
                status="failed",
                error_message=str(exc)[:1000],
            ))
        status = "failed"
        ok = False
        error = str(exc)

    db.flush()
    return {
        "ok": ok,
        "document_id": document.id,
        "status": status,
        "chunk_count": len(created_chunks),
        "error": error,
    }


def index_knowledge_base(
    knowledge_base_id: int,
    db: Session,
    model_ref: str | None = None,
    agent_id: int | None = None,
) -> dict:
    docs = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id == knowledge_base_id,
        KnowledgeDocument.active == True,
    ).order_by(KnowledgeDocument.id.asc()).all()
    results = [index_document(doc, db, model_ref=model_ref, agent_id=agent_id) for doc in docs]
    return {
        "ok": all(result.get("ok") for result in results),
        "knowledge_base_id": knowledge_base_id,
        "document_count": len(results),
        "chunk_count": sum(result.get("chunk_count", 0) for result in results),
        "results": results,
    }


def reindex_all(db: Session, model_ref: str | None = None, agent_id: int | None = None) -> dict:
    bases = db.query(KnowledgeBase).filter(KnowledgeBase.active == True).order_by(KnowledgeBase.id.asc()).all()
    results = [index_knowledge_base(kb.id, db, model_ref=model_ref, agent_id=agent_id) for kb in bases]
    return {
        "ok": all(result.get("ok") for result in results),
        "knowledge_base_count": len(results),
        "document_count": sum(result.get("document_count", 0) for result in results),
        "chunk_count": sum(result.get("chunk_count", 0) for result in results),
        "results": results,
    }


def _create_chunk(document: KnowledgeDocument, spec: ChunkSpec, db: Session) -> KnowledgeChunk:
    metadata = dict(spec.metadata)
    metadata["knowledge_base_id"] = document.knowledge_base_id
    metadata["knowledge_base_name"] = document.knowledge_base.name if document.knowledge_base else ""
    metadata["document_id"] = document.id
    chunk = KnowledgeChunk(
        knowledge_base_id=document.knowledge_base_id,
        document_id=document.id,
        chunk_index=spec.chunk_index,
        title_path=spec.title_path,
        content=spec.content,
        content_hash=spec.content_hash,
        token_count=spec.token_count,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
        active=True,
    )
    db.add(chunk)
    return chunk


def _document_index_is_current(
    document: KnowledgeDocument,
    expected_chunks: list[ChunkSpec],
    model_ref: str,
) -> bool:
    active_chunks = [chunk for chunk in document.chunks if chunk.active]
    active_chunks.sort(key=lambda item: item.chunk_index)
    if len(active_chunks) != len(expected_chunks):
        return False
    for chunk, expected in zip(active_chunks, expected_chunks):
        if chunk.content_hash != expected.content_hash:
            return False
        ready = [
            embedding for embedding in chunk.embeddings
            if embedding.embedding_model == model_ref
            and embedding.content_hash == chunk.content_hash
            and embedding.status == "ready"
        ]
        if not ready:
            return False
    return True


def _store_pgvector(embedding_id: int, vector: list[float], db: Session) -> None:
    bind = db.get_bind()
    if bind.dialect.name != "postgresql" or not _has_embedding_vector_column(db):
        return
    vector_text = "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"
    db.execute(
        text(
            "UPDATE knowledge_embeddings "
            "SET embedding_vector = CAST(:vector_text AS vector) "
            "WHERE id = :embedding_id"
        ),
        {"vector_text": vector_text, "embedding_id": embedding_id},
    )


def _has_embedding_vector_column(db: Session) -> bool:
    try:
        inspector = inspect(db.get_bind())
        return "embedding_vector" in {column["name"] for column in inspector.get_columns("knowledge_embeddings")}
    except Exception:
        return False


def _log_embedding_usage(
    db: Session,
    model_ref: str,
    chunks: list[KnowledgeChunk],
    success: bool,
    latency_ms: int,
    operation: str,
    error: Exception | None = None,
    agent_id: int | None = None,
) -> None:
    provider = parse_model_ref(model_ref)[0]
    if provider == "local":
        return
    prompt_tokens = sum(chunk.token_count or 0 for chunk in chunks)
    add_llm_usage_log(
        db,
        model_ref=model_ref,
        success=success,
        source="rag",
        operation=operation,
        latency_ms=latency_ms,
        error=error,
        agent_id=agent_id,
        usage={
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
