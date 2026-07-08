from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import bindparam, inspect, text
from sqlalchemy.orm import Session

from app.llm import parse_model_ref
from app.llm_usage import add_llm_usage_log
from app.models import (
    Agent,
    AgentKnowledgeBase,
    Conversation,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeEmbedding,
    RagQueryLog,
)
from app.rag.embeddings import _expand_tokens, cosine_similarity, embed_text, embedding_model


DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("RAG_RETRIEVAL_TOP_K", "20"))
DEFAULT_FINAL_CHUNKS = int(os.getenv("RAG_FINAL_CHUNKS", "6"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    knowledge_base_id: int
    document_id: int
    title_path: str
    content: str
    score: float
    vector_score: float
    keyword_score: float


@dataclass(frozen=True)
class RagResult:
    query: str
    rewritten_query: str
    chunks: list[RetrievedChunk]
    context: str
    retrieval_latency_ms: int


def retrieve_agent_context(
    agent: Agent,
    query: str,
    db: Session,
    external_user_id: str = "",
    top_k: int | None = None,
    final_chunks: int | None = None,
    max_context_chars: int | None = None,
    min_score: float | None = None,
    model_ref: str | None = None,
    hybrid_search: bool | None = None,
    rerank_enabled: bool | None = None,
    query_rewrite_enabled: bool | None = None,
) -> RagResult:
    started = time.perf_counter()
    model_ref = model_ref or getattr(agent, "rag_embedding_model", None) or embedding_model()
    top_k = int(top_k or getattr(agent, "rag_top_k", None) or DEFAULT_RETRIEVAL_TOP_K)
    final_chunks = int(final_chunks or getattr(agent, "rag_final_chunks", None) or DEFAULT_FINAL_CHUNKS)
    max_context_chars = int(max_context_chars or getattr(agent, "rag_max_context_chars", None) or DEFAULT_MAX_CONTEXT_CHARS)
    if hybrid_search is None:
        hybrid_search = bool(getattr(agent, "rag_hybrid_search", True))
    if rerank_enabled is None:
        rerank_enabled = bool(getattr(agent, "rag_rerank_enabled", False))
    if query_rewrite_enabled is None:
        query_rewrite_enabled = bool(getattr(agent, "rag_query_rewrite_enabled", False))
    min_score = _default_min_score(model_ref) if min_score is None else min_score
    scoped_kb_ids = _agent_knowledge_base_ids(agent, db)
    if not scoped_kb_ids:
        result = RagResult(query=query, rewritten_query="", chunks=[], context="", retrieval_latency_ms=0)
        _log_query(agent, external_user_id, query, result, db)
        return result

    rewritten_query = _rewrite_query(agent, query, db, external_user_id) if query_rewrite_enabled else ""
    search_query = rewritten_query or query
    query_vector = _embed_query_with_log(agent, search_query, model_ref, db)
    query_terms = set(_terms(search_query))
    rows, pgvector_scores = _candidate_rows(db, scoped_kb_ids, model_ref, query_vector, top_k)

    scored = []
    for chunk, embedding in rows:
        vector_score = pgvector_scores.get(chunk.id)
        if vector_score is None:
            try:
                vector = json.loads(embedding.vector_json or "[]")
            except json.JSONDecodeError:
                vector = []
            vector_score = cosine_similarity(query_vector, vector)
        keyword_score = _keyword_score(query_terms, chunk.content) if hybrid_search else 0.0
        score = ((vector_score * 0.78) + (keyword_score * 0.22)) if hybrid_search else vector_score
        if score >= min_score:
            scored.append(RetrievedChunk(
                chunk_id=chunk.id,
                knowledge_base_id=chunk.knowledge_base_id,
                document_id=chunk.document_id,
                title_path=chunk.title_path or "",
                content=chunk.content or "",
                score=score,
                vector_score=vector_score,
                keyword_score=keyword_score,
            ))

    selected = sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
    if rerank_enabled:
        selected = _rerank(selected, search_query)
    final = _diversify(selected, final_chunks)
    context = build_context(final, max_chars=max_context_chars)
    latency_ms = int((time.perf_counter() - started) * 1000)
    result = RagResult(
        query=query,
        rewritten_query=rewritten_query,
        chunks=final,
        context=context,
        retrieval_latency_ms=latency_ms,
    )
    _log_query(agent, external_user_id, query, result, db)
    return result


def build_context(chunks: list[RetrievedChunk], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
    parts = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"[Kaynak {index}]\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"baslik: {chunk.title_path}\n"
            f"skor: {chunk.score:.3f}\n"
        )
        body = f"{header}{chunk.content.strip()}\n"
        if used and used + len(body) > max_chars:
            break
        parts.append(body)
        used += len(body)
    return "\n---\n".join(parts).strip()


def _agent_knowledge_base_ids(agent: Agent, db: Session) -> list[int]:
    links = db.query(AgentKnowledgeBase).join(
        KnowledgeBase,
        KnowledgeBase.id == AgentKnowledgeBase.knowledge_base_id,
    ).filter(
        AgentKnowledgeBase.agent_id == agent.id,
        AgentKnowledgeBase.active == True,
        KnowledgeBase.active == True,
    ).order_by(AgentKnowledgeBase.priority.asc(), AgentKnowledgeBase.id.asc()).all()
    return [link.knowledge_base_id for link in links]


def _candidate_rows(
    db: Session,
    scoped_kb_ids: list[int],
    model_ref: str,
    query_vector: list[float],
    top_k: int,
) -> tuple[list[tuple[KnowledgeChunk, KnowledgeEmbedding]], dict[int, float]]:
    pgvector_scores = _pgvector_candidate_scores(db, scoped_kb_ids, model_ref, query_vector, top_k)
    query = db.query(KnowledgeChunk, KnowledgeEmbedding).join(
        KnowledgeEmbedding,
        KnowledgeEmbedding.chunk_id == KnowledgeChunk.id,
    ).filter(
        KnowledgeChunk.knowledge_base_id.in_(scoped_kb_ids),
        KnowledgeChunk.active == True,
        KnowledgeEmbedding.embedding_model == model_ref,
        KnowledgeEmbedding.status == "ready",
    )
    if pgvector_scores:
        query = query.filter(KnowledgeChunk.id.in_(list(pgvector_scores.keys())))
    return query.all(), pgvector_scores


def _pgvector_candidate_scores(
    db: Session,
    scoped_kb_ids: list[int],
    model_ref: str,
    query_vector: list[float],
    top_k: int,
) -> dict[int, float]:
    if db.get_bind().dialect.name != "postgresql" or not _has_embedding_vector_column(db):
        return {}
    vector_text = "[" + ",".join(f"{float(value):.8f}" for value in query_vector) + "]"
    statement = text(
        """
        SELECT kc.id AS chunk_id,
               1 - (ke.embedding_vector <=> CAST(:query_vector AS vector)) AS vector_score
        FROM knowledge_chunks kc
        JOIN knowledge_embeddings ke ON ke.chunk_id = kc.id
        WHERE kc.knowledge_base_id IN :kb_ids
          AND kc.active = TRUE
          AND ke.embedding_model = :model_ref
          AND ke.status = 'ready'
          AND ke.embedding_vector IS NOT NULL
        ORDER BY ke.embedding_vector <=> CAST(:query_vector AS vector)
        LIMIT :limit
        """
    ).bindparams(bindparam("kb_ids", expanding=True))
    try:
        rows = db.execute(statement, {
            "kb_ids": scoped_kb_ids,
            "model_ref": model_ref,
            "query_vector": vector_text,
            "limit": max(top_k * 4, top_k),
        }).mappings().all()
        return {int(row["chunk_id"]): float(row["vector_score"] or 0.0) for row in rows}
    except Exception as exc:
        print(f"⚠️ pgvector retrieval fallback: {exc}")
        return {}


def _has_embedding_vector_column(db: Session) -> bool:
    try:
        inspector = inspect(db.get_bind())
        return "embedding_vector" in {column["name"] for column in inspector.get_columns("knowledge_embeddings")}
    except Exception:
        return False


def _diversify(chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    per_document: dict[int, int] = {}
    for chunk in chunks:
        if len(selected) >= limit:
            break
        if per_document.get(chunk.document_id, 0) >= 3 and len(chunks) > limit:
            continue
        selected.append(chunk)
        per_document[chunk.document_id] = per_document.get(chunk.document_id, 0) + 1
    if len(selected) < limit:
        seen = {chunk.chunk_id for chunk in selected}
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            selected.append(chunk)
            if len(selected) >= limit:
                break
    return selected


def _rerank(chunks: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
    query_terms = set(_terms(query))
    query_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", query or ""))
    reranked = []
    for chunk in chunks:
        title_terms = set(_terms(chunk.title_path))
        content = chunk.content.lower()
        phrase_bonus = 0.08 if query.lower().strip() and query.lower().strip() in content else 0.0
        title_bonus = min(0.08, 0.02 * len(query_terms & title_terms))
        number_bonus = 0.05 if query_numbers and query_numbers & set(re.findall(r"\d+(?:[.,]\d+)?", chunk.content)) else 0.0
        score = chunk.score + phrase_bonus + title_bonus + number_bonus
        reranked.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            knowledge_base_id=chunk.knowledge_base_id,
            document_id=chunk.document_id,
            title_path=chunk.title_path,
            content=chunk.content,
            score=score,
            vector_score=chunk.vector_score,
            keyword_score=chunk.keyword_score,
        ))
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def _keyword_score(query_terms: set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = set(_terms(content))
    if not content_terms:
        return 0.0
    overlap = len(query_terms & content_terms)
    return min(1.0, overlap / max(1, len(query_terms)))


def _terms(text: str) -> list[str]:
    terms = re.findall(r"[\wğüşöçıİĞÜŞÖÇ]+", (text or "").lower(), flags=re.UNICODE)
    stopwords = {
        "bir", "bu", "şu", "ve", "ile", "için", "mi", "mı", "mu", "mü",
        "ne", "nasıl", "var", "yok", "kaç", "hangi", "ben", "sen", "siz",
    }
    return _expand_tokens([term for term in terms if len(term) > 1 and term not in stopwords])


def _default_min_score(model_ref: str) -> float:
    if model_ref.startswith("local:"):
        return float(os.getenv("RAG_MIN_SCORE", "0.04"))
    return float(os.getenv("RAG_MIN_SCORE", "0.55"))


def _rewrite_query(agent: Agent, query: str, db: Session, external_user_id: str) -> str:
    if not external_user_id:
        return query
    messages = db.query(Conversation).filter(
        Conversation.agent_id == agent.id,
        Conversation.external_user_id == external_user_id,
        Conversation.role == "user",
    ).order_by(Conversation.created_at.desc()).limit(6).all()
    previous = [message.message for message in reversed(messages) if message.message and message.message != query]
    if not previous:
        return query
    context = " ".join(previous)[-700:]
    return f"Konusma baglami: {context}\nSon soru: {query}"


def _log_query(agent: Agent, external_user_id: str, query: str, result: RagResult, db: Session) -> None:
    try:
        db.add(RagQueryLog(
            agent_id=agent.id if agent else None,
            external_user_id=external_user_id or "",
            query=query,
            rewritten_query=result.rewritten_query,
            retrieved_chunk_ids_json=json.dumps([chunk.chunk_id for chunk in result.chunks]),
            scores_json=json.dumps([
                {
                    "chunk_id": chunk.chunk_id,
                    "score": round(chunk.score, 6),
                    "vector_score": round(chunk.vector_score, 6),
                    "keyword_score": round(chunk.keyword_score, 6),
                }
                for chunk in result.chunks
            ]),
            selected_context_chars=len(result.context or ""),
            retrieval_latency_ms=result.retrieval_latency_ms,
            rerank_latency_ms=0,
            created_at=datetime.now(timezone.utc),
        ))
        db.flush()
    except Exception as exc:
        print(f"⚠️ RAG query log hatası: {exc}")


def _embed_query_with_log(agent: Agent, query: str, model_ref: str, db: Session) -> list[float]:
    started = time.perf_counter()
    try:
        vector = embed_text(query, model_ref=model_ref)
        _log_query_embedding_usage(db, agent, model_ref, query, True, int((time.perf_counter() - started) * 1000))
        return vector
    except Exception as exc:
        _log_query_embedding_usage(db, agent, model_ref, query, False, int((time.perf_counter() - started) * 1000), exc)
        raise


def _log_query_embedding_usage(
    db: Session,
    agent: Agent,
    model_ref: str,
    query: str,
    success: bool,
    latency_ms: int,
    error: Exception | None = None,
) -> None:
    provider = parse_model_ref(model_ref)[0]
    if provider == "local":
        return
    prompt_tokens = max(1, int(len((query or "").split()) * 1.25))
    try:
        add_llm_usage_log(
            db,
            model_ref=model_ref,
            success=success,
            source="rag",
            operation="embedding:query",
            latency_ms=latency_ms,
            error=error,
            agent_id=agent.id if agent else None,
            usage={
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        )
        db.flush()
    except Exception as log_exc:
        print(f"⚠️ RAG embedding usage log hatası: {log_exc}")
