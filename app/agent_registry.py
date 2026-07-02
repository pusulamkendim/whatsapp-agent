from datetime import datetime, timedelta, timezone
import time

from sqlalchemy.orm import Session
from app.models import Agent, AgentKnowledgeBase, KnowledgeDocument, LlmModel, LlmUsageLog
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
        knowledge = build_agent_knowledge(agent, db)
        response = retreat_chat(customer_id, message, knowledge_base=knowledge or None, model=model)
        _log_llm_usage(db, agent, model, success=True, latency_ms=_elapsed_ms(started), usage=summarize_llm_usage(usage_events))
        return response

    if agent.type == "generic_prompt":
        knowledge = build_agent_knowledge(agent, db)
        response = _generic_prompt_response(agent, customer_id, message, knowledge, model)
        _log_llm_usage(db, agent, model, success=True, latency_ms=_elapsed_ms(started), usage=summarize_llm_usage(usage_events))
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
    estimated_cost = _estimate_llm_cost(db, model_ref, usage)
    try:
        db.add(LlmUsageLog(
            agent_id=agent.id,
            provider=provider,
            model_ref=model_ref,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            estimated_cost=estimated_cost,
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
    history = generic_conversations[conversation_key]
    history.append({"role": "user", "content": message})
    text = run_openai_simple_chat(model, history)
    if len(history) > 32:
        generic_conversations[conversation_key] = [history[0], *history[-31:]]
    return text
