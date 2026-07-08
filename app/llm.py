import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

import requests
from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY


GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
_USAGE_EVENTS: ContextVar[list[dict] | None] = ContextVar("llm_usage_events", default=None)

MODEL_OPTIONS = [
    {
        "provider": "gemini",
        "label": "Gemini 2.5 Flash",
        "model": "gemini:gemini-2.5-flash",
        "env": "GEMINI_API_KEY",
        "notes": "Mevcut varsayilan model",
    },
    {
        "provider": "gemini",
        "label": "Gemini 2.5 Pro",
        "model": "gemini:gemini-2.5-pro",
        "env": "GEMINI_API_KEY",
        "notes": "Daha yuksek kalite/reasoning; OCR duzeltme icin kullanilabilir",
    },
    {
        "provider": "gemini",
        "label": "Gemini 2.0 Flash",
        "model": "gemini:gemini-2.0-flash",
        "env": "GEMINI_API_KEY",
        "notes": "Hizli multimodal model",
    },
    {
        "provider": "gemini",
        "label": "Gemini 1.5 Flash",
        "model": "gemini:gemini-1.5-flash",
        "env": "GEMINI_API_KEY",
        "notes": "Geriye uyumlu Gemini secenegi",
    },
    {
        "provider": "deepseek",
        "label": "DeepSeek V4 Flash",
        "model": "deepseek:deepseek-v4-flash",
        "env": "DEEPSEEK_API_KEY",
        "notes": "OpenAI-compatible DeepSeek API",
    },
    {
        "provider": "deepseek",
        "label": "DeepSeek V4 Pro",
        "model": "deepseek:deepseek-v4-pro",
        "env": "DEEPSEEK_API_KEY",
        "notes": "Reasoning/pro model",
    },
    {
        "provider": "openrouter",
        "label": "OpenRouter Auto",
        "model": "openrouter:openrouter/auto",
        "env": "OPENROUTER_API_KEY",
        "notes": "OpenRouter model router",
    },
    {
        "provider": "openrouter",
        "label": "OpenRouter GPT OSS 20B Free",
        "model": "openrouter:openai/gpt-oss-20b:free",
        "env": "OPENROUTER_API_KEY",
        "notes": "OpenRouter free variant; availability can change",
    },
    {
        "provider": "openrouter",
        "label": "OpenRouter Llama 3.3 70B Free",
        "model": "openrouter:meta-llama/llama-3.3-70b-instruct:free",
        "env": "OPENROUTER_API_KEY",
        "notes": "OpenRouter free variant; availability can change",
    },
    {
        "provider": "ollama",
        "label": "Ollama Qwen 2.5 7B",
        "model": "ollama:qwen2.5:7b",
        "env": "OLLAMA_BASE_URL",
        "notes": "Local/free if Ollama is running",
    },
]


def parse_model_ref(model_ref: str | None) -> tuple[str, str]:
    value = (model_ref or "gemini:gemini-2.5-flash").strip()
    for provider in ["gemini", "deepseek", "openrouter", "ollama", "openai"]:
        prefix = f"{provider}:"
        if value.startswith(prefix):
            return provider, value[len(prefix):]
    if ":" in value:
        provider, model = value.split(":", 1)
        if provider and model:
            return provider, model
    return "gemini", value


def provider_label(model_ref: str | None) -> str:
    provider, model = parse_model_ref(model_ref)
    return f"{provider}:{model}"


def is_gemini_model(model_ref: str | None) -> bool:
    return parse_model_ref(model_ref)[0] == "gemini"


@contextmanager
def capture_llm_usage():
    events: list[dict] = []
    token = _USAGE_EVENTS.set(events)
    try:
        yield events
    finally:
        _USAGE_EVENTS.reset(token)


def summarize_llm_usage(events: list[dict]) -> dict:
    return {
        "prompt_tokens": _sum_usage(events, "prompt_tokens"),
        "completion_tokens": _sum_usage(events, "completion_tokens"),
        "total_tokens": _sum_usage(events, "total_tokens"),
        "actual_cost": _sum_usage(events, "actual_cost"),
        "generation_id": _join_unique(events, "generation_id"),
        "actual_model_ref": _join_unique(events, "actual_model_ref"),
        "actual_provider": _join_unique(events, "actual_provider"),
        "router": _join_unique(events, "router"),
        "cost_details_json": _join_unique(events, "cost_details_json", limit=4000),
    }


def record_gemini_usage(model_ref: str, response, source: str | None = None, operation: str | None = None):
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    record_llm_usage(
        model_ref=model_ref,
        prompt_tokens=getattr(usage, "prompt_token_count", None),
        completion_tokens=getattr(usage, "candidates_token_count", None),
        total_tokens=getattr(usage, "total_token_count", None),
        source=source,
        operation=operation,
    )


def record_llm_usage(
    model_ref: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    actual_cost: float | None = None,
    generation_id: str | None = None,
    actual_model_ref: str | None = None,
    actual_provider: str | None = None,
    router: str | None = None,
    cost_details_json: str | None = None,
    source: str | None = None,
    operation: str | None = None,
):
    events = _USAGE_EVENTS.get()
    if events is None:
        return
    events.append({
        "model_ref": model_ref,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "actual_cost": actual_cost,
        "generation_id": generation_id,
        "actual_model_ref": actual_model_ref,
        "actual_provider": actual_provider,
        "router": router,
        "cost_details_json": cost_details_json,
        "source": source,
        "operation": operation,
    })


def _sum_usage(events: list[dict], key: str) -> int | None:
    values = [event.get(key) for event in events if event.get(key) is not None]
    return sum(values) if values else None


def _join_unique(events: list[dict], key: str, limit: int = 500) -> str:
    values = []
    for event in events:
        value = event.get(key)
        if value and value not in values:
            values.append(str(value))
    return " | ".join(values)[:limit]


def openai_compatible_chat(
    model_ref: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
) -> dict:
    provider, model = parse_model_ref(model_ref)
    base_url, api_key, extra_headers = _openai_compatible_config(provider)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {"Content-Type": "application/json", **extra_headers}
    if provider == "openrouter":
        headers["X-OpenRouter-Metadata"] = "enabled"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = _post_with_transient_retry(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        payload=payload,
    )
    if not response.ok:
        detail = response.text[:500] if response.text else response.reason
        raise RuntimeError(f"{provider}:{model} HTTP {response.status_code}: {detail}")
    data = response.json()
    usage = data.get("usage") or {}
    openrouter_metadata = data.get("openrouter_metadata") or {}
    selected_endpoint = _selected_openrouter_endpoint(openrouter_metadata)
    cost_details = usage.get("cost_details") or {}
    record_llm_usage(
        model_ref=model_ref,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        actual_cost=usage.get("cost"),
        generation_id=data.get("id"),
        actual_model_ref=data.get("model") if provider == "openrouter" else None,
        actual_provider=selected_endpoint.get("provider") if selected_endpoint else None,
        router=openrouter_metadata.get("requested") or (model if provider == "openrouter" else None),
        cost_details_json=json.dumps(cost_details)[:4000] if cost_details else "",
    )
    return data["choices"][0]["message"]


def _selected_openrouter_endpoint(metadata: dict) -> dict:
    endpoints = ((metadata.get("endpoints") or {}).get("available") or [])
    for endpoint in endpoints:
        if endpoint.get("selected"):
            return endpoint
    attempts = metadata.get("attempts") or []
    for attempt in reversed(attempts):
        if attempt.get("status") == 200:
            return attempt
    return {}


def _post_with_transient_retry(url: str, headers: dict, payload: dict) -> requests.Response:
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code not in {429, 502, 503, 504}:
        return response

    retry_after = _retry_after_seconds(response)
    if retry_after is None or retry_after > 10:
        return response

    time.sleep(retry_after)
    return requests.post(url, headers=headers, json=payload, timeout=60)


def _retry_after_seconds(response: requests.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return max(0, float(header))
        except ValueError:
            return None
    try:
        data = response.json()
    except ValueError:
        return None
    metadata = ((data.get("error") or {}).get("metadata") or {})
    value = metadata.get("retry_after_seconds") or metadata.get("retry_after_seconds_raw")
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        return None


def run_openai_tool_loop(
    model_ref: str,
    messages: list[dict],
    tools: list[dict],
    execute_tool: Callable[[str, dict], str],
    max_iterations: int = 4,
    temperature: float = 0.7,
) -> str:
    for _ in range(max_iterations):
        assistant_message = openai_compatible_chat(model_ref, messages, tools, temperature)
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            content = assistant_message.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            return content

        messages.append({
            "role": "assistant",
            "content": assistant_message.get("content"),
            "tool_calls": tool_calls,
        })
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            name = function.get("name") or ""
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": name,
                "content": result,
            })
    return "Bu isteği tamamlarken araç çağrısı sınırına ulaştım."


def run_openai_simple_chat(
    model_ref: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> str:
    assistant_message = openai_compatible_chat(model_ref, messages, None, temperature)
    content = assistant_message.get("content") or ""
    messages.append({"role": "assistant", "content": content})
    return content


def gemini_tool_from_openai_tools(openai_tools: list[dict]) -> types.Tool:
    declarations = []
    for tool in openai_tools:
        fn = tool["function"]
        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=_schema_to_gemini(fn.get("parameters") or {"type": "object", "properties": {}}),
        ))
    return types.Tool(function_declarations=declarations)


def _schema_to_gemini(schema: dict) -> types.Schema:
    schema_type = (schema.get("type") or "object").upper()
    type_map = {
        "OBJECT": types.Type.OBJECT,
        "STRING": types.Type.STRING,
        "INTEGER": types.Type.INTEGER,
        "NUMBER": types.Type.NUMBER,
        "BOOLEAN": types.Type.BOOLEAN,
        "ARRAY": types.Type.ARRAY,
    }
    kwargs = {
        "type": type_map.get(schema_type, types.Type.STRING),
        "description": schema.get("description"),
    }
    properties = schema.get("properties") or {}
    if properties:
        kwargs["properties"] = {key: _schema_to_gemini(value) for key, value in properties.items()}
    if schema.get("required"):
        kwargs["required"] = schema["required"]
    if schema.get("items"):
        kwargs["items"] = _schema_to_gemini(schema["items"])
    return types.Schema(**{key: value for key, value in kwargs.items() if value is not None})


def _openai_compatible_config(provider: str) -> tuple[str, str, dict]:
    db_config = _provider_config_from_db(provider)
    if provider == "deepseek":
        base_url, api_key_env = db_config or ("https://api.deepseek.com", "DEEPSEEK_API_KEY")
        return (
            os.getenv("DEEPSEEK_BASE_URL", base_url),
            os.getenv(api_key_env, ""),
            {},
        )
    if provider == "openrouter":
        base_url, api_key_env = db_config or ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
        return (
            os.getenv("OPENROUTER_BASE_URL", base_url),
            os.getenv(api_key_env, ""),
            {
                "HTTP-Referer": os.getenv("BASE_URL", "https://agentapi.pusulamkendim.com"),
                "X-OpenRouter-Title": "AgentLense",
            },
        )
    if provider == "ollama":
        base_url, api_key_env = db_config or ("http://localhost:11434/v1", "OLLAMA_API_KEY")
        return (
            os.getenv("OLLAMA_BASE_URL", base_url),
            os.getenv(api_key_env, "ollama"),
            {},
        )
    if provider == "openai":
        base_url, api_key_env = db_config or ("https://api.openai.com/v1", "OPENAI_API_KEY")
        return (
            os.getenv("OPENAI_BASE_URL", base_url),
            os.getenv(api_key_env, ""),
            {},
        )
    if db_config:
        base_url, api_key_env = db_config
        return (base_url, os.getenv(api_key_env, ""), {})
    raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")


def _provider_config_from_db(provider: str) -> tuple[str, str] | None:
    try:
        from app.database import SessionLocal
        from app.models import LlmProvider

        db = SessionLocal()
        try:
            row = db.query(LlmProvider).filter(LlmProvider.slug == provider, LlmProvider.active == True).first()
            if row and row.base_url:
                return row.base_url, row.api_key_env or ""
        finally:
            db.close()
    except Exception:
        return None
    return None
