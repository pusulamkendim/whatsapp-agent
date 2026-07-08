from sqlalchemy.orm import Session

from app.llm import parse_model_ref
from app.models import LlmModel, LlmUsageLog


def add_llm_usage_log(
    db: Session,
    model_ref: str,
    success: bool,
    source: str = "",
    operation: str = "",
    latency_ms: int | None = None,
    error: Exception | None = None,
    usage: dict | None = None,
    agent_id: int | None = None,
):
    usage = usage or {}
    normalized_ref = normalize_model_ref(model_ref or usage.get("model_ref") or "")
    provider = parse_model_ref(normalized_ref)[0]
    actual_cost = usage.get("actual_cost")
    estimated_cost = actual_cost if actual_cost is not None else estimate_llm_cost(db, normalized_ref, usage)
    db.add(LlmUsageLog(
        agent_id=agent_id,
        source=source,
        operation=operation,
        provider=provider,
        model_ref=normalized_ref,
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


def add_llm_usage_events(
    db: Session,
    events: list[dict],
    source: str,
    operation: str,
    latency_ms: int | None = None,
    success: bool = True,
    error: Exception | None = None,
):
    for event in events:
        add_llm_usage_log(
            db,
            model_ref=event.get("model_ref") or "",
            success=success,
            source=event.get("source") or source,
            operation=event.get("operation") or operation,
            latency_ms=latency_ms,
            error=error,
            usage=event,
        )


def estimate_llm_cost(db: Session, model_ref: str, usage: dict) -> float | None:
    model = db.query(LlmModel).filter(LlmModel.model_ref == normalize_model_ref(model_ref)).first()
    if not model or (model.input_price is None and model.output_price is None):
        return None

    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    input_price = model.input_price or 0
    output_price = model.output_price or 0
    return ((prompt_tokens * input_price) + (completion_tokens * output_price)) / 1_000_000


def normalize_model_ref(model_ref: str | None) -> str:
    provider, model = parse_model_ref(model_ref)
    return f"{provider}:{model}"
