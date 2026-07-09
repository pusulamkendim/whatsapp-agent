from datetime import datetime, timedelta, timezone
import time

from sqlalchemy.orm import Session
from app.models import Agent, AgentKnowledgeBase, KnowledgeChunk, KnowledgeDocument, LlmModel, LlmUsageLog, RagQueryLog
from app.rag.indexing import index_knowledge_base
from app.rag.retrieval import retrieve_agent_context
from app.agent import chat as restaurant_chat
from app.retreat_agent import chat as retreat_chat
from app.llm import (
    GEMINI_CLIENT,
    capture_llm_usage,
    is_gemini_model,
    parse_model_ref,
    record_gemini_usage,
    run_openai_simple_chat,
    summarize_llm_usage,
)
from google.genai import types


RESTAURANT_ID = 1
RESTAURANT_NAME = "Lezzet Durağı"


def run_agent(agent: Agent, customer_id: str, message: str, db: Session) -> str:
    primary_model = agent.model or "gemini:gemini-2.5-flash"
    fallback_model = agent.fallback_model or ""
    try:
        return _run_agent_with_model(agent, customer_id, message, db, primary_model)
    except Exception as exc:
        _log_llm_usage(db, agent, primary_model, success=False, error=exc)
        if agent.failover_enabled and fallback_model and fallback_model != primary_model:
            try:
                return _run_agent_with_model(agent, customer_id, message, db, fallback_model)
            except Exception as fallback_exc:
                _log_llm_usage(db, agent, fallback_model, success=False, error=fallback_exc)
                raise fallback_exc
        raise exc


def _run_agent_with_model(agent: Agent, customer_id: str, message: str, db: Session, model: str) -> str:
    rate_error = _rate_limit_error(db, model)
    if rate_error:
        raise RuntimeError(rate_error)

    started = time.perf_counter()
    with capture_llm_usage() as usage_events:
        return _run_agent_with_usage(agent, customer_id, message, db, model, started, usage_events)


def _run_agent_with_usage(
    agent: Agent,
    customer_id: str,
    message: str,
    db: Session,
    model: str,
    started: float,
    usage_events: list[dict],
) -> str:
    if agent.type == "restaurant" or agent.slug == "restaurant":
        response = restaurant_chat(customer_id, message, RESTAURANT_ID, RESTAURANT_NAME, db, model=model)
        _log_llm_usage(db, agent, model, success=True, latency_ms=_elapsed_ms(started), usage=summarize_llm_usage(usage_events))
        return response

    if agent.type == "retreat" or agent.slug == "retreat":
        knowledge = build_agent_prompt_knowledge(agent, customer_id, message, db)
        response = retreat_chat(customer_id, message, knowledge_base=knowledge or None, model=model)
        usage = summarize_llm_usage(usage_events)
        latency_ms = _elapsed_ms(started)
        _annotate_latest_rag_log(db, agent, customer_id, model, latency_ms, usage)
        _log_llm_usage(db, agent, model, success=True, latency_ms=latency_ms, usage=usage)
        return response

    if agent.type == "generic_prompt":
        knowledge = build_agent_prompt_knowledge(agent, customer_id, message, db)
        response = _generic_prompt_response(agent, customer_id, message, knowledge, model)
        usage = summarize_llm_usage(usage_events)
        latency_ms = _elapsed_ms(started)
        _annotate_latest_rag_log(db, agent, customer_id, model, latency_ms, usage)
        _log_llm_usage(db, agent, model, success=True, latency_ms=latency_ms, usage=usage)
        return response

    return "Bu agent tipi henüz desteklenmiyor."


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log_llm_usage(
    db: Session,
    agent: Agent,
    model_ref: str,
    success: bool,
    latency_ms: int | None = None,
    error: Exception | None = None,
    usage: dict | None = None,
):
    provider = parse_model_ref(model_ref)[0]
    usage = usage or {}
    actual_cost = usage.get("actual_cost")
    estimated_cost = actual_cost if actual_cost is not None else _estimate_llm_cost(db, model_ref, usage)
    try:
        db.add(LlmUsageLog(
            agent_id=agent.id,
            source="agent",
            operation=agent.slug or agent.type,
            provider=provider,
            model_ref=model_ref,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            generation_id=usage.get("generation_id") or "",
            actual_model_ref=usage.get("actual_model_ref") or "",
            actual_provider=usage.get("actual_provider") or "",
            router=usage.get("router") or "",
            cost_details_json=usage.get("cost_details_json") or "",
            latency_ms=latency_ms,
            success=success,
            error_code=type(error).__name__ if error else "",
            error_message=str(error)[:1000] if error else "",
        ))
        db.commit()
    except Exception as log_exc:
        db.rollback()
        print(f"⚠️ LLM usage log hatası: {log_exc}")


def _estimate_llm_cost(db: Session, model_ref: str, usage: dict) -> float | None:
    model = db.query(LlmModel).filter(LlmModel.model_ref == model_ref).first()
    if not model or (model.input_price is None and model.output_price is None):
        return None

    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    input_price = model.input_price or 0
    output_price = model.output_price or 0
    return ((prompt_tokens * input_price) + (completion_tokens * output_price)) / 1_000_000


def _annotate_latest_rag_log(
    db: Session,
    agent: Agent,
    customer_id: str,
    model_ref: str,
    latency_ms: int,
    usage: dict,
) -> None:
    try:
        log = db.query(RagQueryLog).filter(
            RagQueryLog.agent_id == agent.id,
            RagQueryLog.external_user_id == customer_id,
        ).order_by(RagQueryLog.id.desc()).first()
        if not log:
            return
        log.answer_latency_ms = latency_ms
        log.model_ref = model_ref
        log.prompt_tokens = usage.get("prompt_tokens")
        log.completion_tokens = usage.get("completion_tokens")
        log.total_tokens = usage.get("total_tokens")
        db.flush()
    except Exception as exc:
        print(f"⚠️ RAG log cevap metadatasi yazilamadi: {exc}")


def _rate_limit_error(db: Session, model_ref: str) -> str | None:
    model = db.query(LlmModel).filter(LlmModel.model_ref == model_ref, LlmModel.active == True).first()
    if not model or not model.rate_limit_rpm:
        return None
    since = datetime.now(timezone.utc) - timedelta(minutes=1)
    calls = db.query(LlmUsageLog).filter(
        LlmUsageLog.model_ref == model_ref,
        LlmUsageLog.created_at >= since,
    ).count()
    if calls >= model.rate_limit_rpm:
        return f"rate_limit_rpm_exceeded:{model_ref}"
    return None


def build_agent_knowledge(agent: Agent, db: Session) -> str:
    links = db.query(AgentKnowledgeBase).filter(
        AgentKnowledgeBase.agent_id == agent.id,
        AgentKnowledgeBase.active == True,
    ).order_by(AgentKnowledgeBase.priority.asc(), AgentKnowledgeBase.id.asc()).all()

    sections = []
    for link in links:
        kb = link.knowledge_base
        if not kb or not kb.active:
            continue
        docs = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.knowledge_base_id == kb.id,
            KnowledgeDocument.active == True,
        ).order_by(KnowledgeDocument.id.asc()).all()
        if not docs:
            continue
        doc_parts = [f"# {kb.name}"]
        if kb.description:
            doc_parts.append(kb.description)
        for doc in docs:
            doc_parts.append(f"## {doc.filename}\n{doc.content}")
        sections.append("\n\n".join(doc_parts))

    if agent.knowledge_base:
        sections.append(f"# {agent.name} inline knowledge\n{agent.knowledge_base}")

    return "\n\n---\n\n".join(sections)


def build_agent_prompt_knowledge(agent: Agent, customer_id: str, message: str, db: Session) -> str:
    sections = []
    rag_context = _retrieve_or_warm_agent_context(agent, customer_id, message, db)
    if rag_context:
        sections.append(
            "RAG ILE SECILEN ILGILI BILGI PARCALARI:\n"
            "Sadece bu kaynaklarda yer alan bilgilere dayan. "
            "Cevap icin yeterli bilgi yoksa bunu acikca soyle, tahmin yurutme.\n\n"
            f"{rag_context}"
        )
    if agent.knowledge_base:
        sections.append(f"AGENT SABIT NOTLARI:\n{agent.knowledge_base}")
    return "\n\n---\n\n".join(sections)


def _retrieve_or_warm_agent_context(agent: Agent, customer_id: str, message: str, db: Session) -> str:
    try:
        result = retrieve_agent_context(agent, message, db, external_user_id=customer_id)
        if result.context:
            return result.context

        if not _agent_has_indexable_docs(agent, db):
            return ""

        for link in db.query(AgentKnowledgeBase).filter(
            AgentKnowledgeBase.agent_id == agent.id,
            AgentKnowledgeBase.active == True,
        ).order_by(AgentKnowledgeBase.priority.asc(), AgentKnowledgeBase.id.asc()).all():
            index_knowledge_base(
                link.knowledge_base_id,
                db,
                model_ref=agent.rag_embedding_model or None,
                agent_id=agent.id,
            )
        db.flush()

        result = retrieve_agent_context(agent, message, db, external_user_id=customer_id)
        return result.context
    except Exception as exc:
        print(f"⚠️ RAG retrieval atlandı: {exc}")
        return ""


def _agent_has_indexable_docs(agent: Agent, db: Session) -> bool:
    links = db.query(AgentKnowledgeBase).filter(
        AgentKnowledgeBase.agent_id == agent.id,
        AgentKnowledgeBase.active == True,
    ).all()
    if not links:
        return False
    kb_ids = [link.knowledge_base_id for link in links]
    active_docs = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.knowledge_base_id.in_(kb_ids),
        KnowledgeDocument.active == True,
    ).count()
    if active_docs == 0:
        return False
    active_chunks = db.query(KnowledgeChunk).filter(
        KnowledgeChunk.knowledge_base_id.in_(kb_ids),
        KnowledgeChunk.active == True,
    ).count()
    return active_chunks == 0


generic_conversations: dict[str, list[dict]] = {}
generic_gemini_conversations: dict[str, list] = {}


def _generic_prompt_response(agent: Agent, customer_id: str, message: str, knowledge: str, model: str) -> str:
    prompt = agent.system_prompt or f"Sen {agent.name} isimli yardımcı bir asistansın."
    if knowledge:
        prompt = f"{prompt}\n\nBILGI BANKASI:\n{knowledge}"

    conversation_key = f"{agent.id}:{customer_id}"
    if is_gemini_model(model):
        if conversation_key not in generic_gemini_conversations:
            generic_gemini_conversations[conversation_key] = []
        history = generic_gemini_conversations[conversation_key]
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        response = GEMINI_CLIENT.models.generate_content(
            model=parse_model_ref(model)[1],
            contents=history,
            config=types.GenerateContentConfig(system_instruction=prompt, temperature=0.7),
        )
        record_gemini_usage(model, response)
        text = response.candidates[0].content.parts[0].text
        history.append(response.candidates[0].content)
        if len(history) > 30:
            generic_gemini_conversations[conversation_key] = history[-30:]
        return text

    if conversation_key not in generic_conversations:
        generic_conversations[conversation_key] = [{"role": "system", "content": prompt}]
    else:
        generic_conversations[conversation_key][0] = {"role": "system", "content": prompt}
    history = generic_conversations[conversation_key]
    history.append({"role": "user", "content": message})
    text = run_openai_simple_chat(model, history)
    if len(history) > 32:
        generic_conversations[conversation_key] = [history[0], *history[-31:]]
    return text
