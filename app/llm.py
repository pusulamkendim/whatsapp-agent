import json
import os
from typing import Callable

import requests
from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY


GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

MODEL_OPTIONS = [
    {
        "provider": "gemini",
        "label": "Gemini 2.5 Flash",
        "model": "gemini:gemini-2.5-flash",
        "env": "GEMINI_API_KEY",
        "notes": "Mevcut varsayilan model",
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
    return "gemini", value


def provider_label(model_ref: str | None) -> str:
    provider, model = parse_model_ref(model_ref)
    return f"{provider}:{model}"


def is_gemini_model(model_ref: str | None) -> bool:
    return parse_model_ref(model_ref)[0] == "gemini"


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
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if not response.ok:
        detail = response.text[:500] if response.text else response.reason
        raise RuntimeError(f"{provider}:{model} HTTP {response.status_code}: {detail}")
    data = response.json()
    return data["choices"][0]["message"]


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
    if provider == "deepseek":
        return (
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            os.getenv("DEEPSEEK_API_KEY", ""),
            {},
        )
    if provider == "openrouter":
        return (
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            os.getenv("OPENROUTER_API_KEY", ""),
            {
                "HTTP-Referer": os.getenv("BASE_URL", "https://agentapi.pusulamkendim.com"),
                "X-OpenRouter-Title": "AgentLense",
            },
        )
    if provider == "ollama":
        return (
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            os.getenv("OLLAMA_API_KEY", "ollama"),
            {},
        )
    if provider == "openai":
        return (
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.getenv("OPENAI_API_KEY", ""),
            {},
        )
    raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")
