from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.llm import parse_model_ref
from app.models import LlmModel


@dataclass(frozen=True)
class ModelPrice:
    input_price: float | None
    output_price: float | None
    source: str


STATIC_PRICE_CATALOG: dict[str, ModelPrice] = {
    "gemini:gemini-2.5-flash": ModelPrice(
        0.30,
        2.50,
        "https://ai.google.dev/gemini-api/docs/pricing",
    ),
    "deepseek:deepseek-v4-flash": ModelPrice(
        0.14,
        0.28,
        "https://api-docs.deepseek.com/quick_start/pricing",
    ),
    "deepseek:deepseek-v4-pro": ModelPrice(
        0.435,
        0.87,
        "https://api-docs.deepseek.com/quick_start/pricing",
    ),
    "openrouter:openai/gpt-oss-20b:free": ModelPrice(
        0,
        0,
        "https://openrouter.ai/openai/gpt-oss-20b:free",
    ),
    "openrouter:meta-llama/llama-3.3-70b-instruct:free": ModelPrice(
        0,
        0,
        "https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free",
    ),
    "ollama:qwen2.5:7b": ModelPrice(
        0,
        0,
        "local_ollama",
    ),
}


def sync_llm_prices(force: bool = False, stale_after_hours: int = 24) -> dict:
    db = SessionLocal()
    try:
        return sync_llm_prices_for_db(db, force=force, stale_after_hours=stale_after_hours)
    finally:
        db.close()


def sync_llm_prices_for_db(db: Session, force: bool = False, stale_after_hours: int = 24) -> dict:
    now = datetime.now(timezone.utc)
    models = db.query(LlmModel).filter(LlmModel.active == True).all()
    openrouter_prices = None
    result = {"updated": 0, "skipped": 0, "errors": 0, "models": []}

    for model in models:
        if not force and not _is_stale(model.pricing_checked_at, now, stale_after_hours):
            result["skipped"] += 1
            continue

        try:
            price = _price_for_model(model, openrouter_prices)
            if price is None and parse_model_ref(model.model_ref)[0] == "openrouter":
                if openrouter_prices is None:
                    openrouter_prices = fetch_openrouter_prices()
                price = _price_for_model(model, openrouter_prices)

            if price is None:
                _mark_model_unpriced(model, now)
                result["skipped"] += 1
                result["models"].append({"model_ref": model.model_ref, "status": "unpriced"})
                continue

            model.input_price = price.input_price
            model.output_price = price.output_price
            model.pricing_source = price.source
            model.pricing_checked_at = now
            model.pricing_sync_error = ""
            result["updated"] += 1
            result["models"].append({"model_ref": model.model_ref, "status": "updated"})
        except Exception as exc:
            model.pricing_checked_at = now
            model.pricing_sync_error = str(exc)[:1000]
            result["errors"] += 1
            result["models"].append({"model_ref": model.model_ref, "status": "error", "error": type(exc).__name__})

    db.commit()
    return result


def fetch_openrouter_prices() -> dict[str, ModelPrice]:
    response = requests.get("https://openrouter.ai/api/v1/models", timeout=12)
    response.raise_for_status()
    prices = {}
    for item in (response.json() or {}).get("data", []):
        model_id = item.get("id")
        pricing = item.get("pricing") or {}
        prompt_price = _openrouter_price_per_million(pricing.get("prompt"))
        completion_price = _openrouter_price_per_million(pricing.get("completion"))
        if model_id and prompt_price is not None and completion_price is not None:
            prices[f"openrouter:{model_id}"] = ModelPrice(
                prompt_price,
                completion_price,
                f"https://openrouter.ai/{model_id}",
            )
    return prices


def _price_for_model(model: LlmModel, openrouter_prices: dict[str, ModelPrice] | None) -> ModelPrice | None:
    if model.model_ref == "openrouter:openrouter/auto":
        return None
    if openrouter_prices and model.model_ref in openrouter_prices:
        return openrouter_prices[model.model_ref]
    return STATIC_PRICE_CATALOG.get(model.model_ref)


def _mark_model_unpriced(model: LlmModel, checked_at: datetime):
    model.pricing_checked_at = checked_at
    if model.model_ref == "openrouter:openrouter/auto":
        model.pricing_source = "actual_cost_from_openrouter_response"
        model.pricing_sync_error = "Dynamic router; fixed price is not available."
    else:
        model.pricing_sync_error = "No pricing source matched this model."


def _openrouter_price_per_million(value) -> float | None:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def _is_stale(checked_at: datetime | None, now: datetime, stale_after_hours: int) -> bool:
    if checked_at is None:
        return True
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    return checked_at <= now - timedelta(hours=stale_after_hours)
