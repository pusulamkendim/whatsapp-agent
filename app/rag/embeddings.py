from __future__ import annotations

import hashlib
import math
import os
import re
import time

import requests

from app.llm import GEMINI_CLIENT, _openai_compatible_config, parse_model_ref


DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "local:hash-v1")
LOCAL_HASH_DIM = 384
DEFAULT_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))
DEFAULT_RETRIES = int(os.getenv("RAG_EMBEDDING_RETRIES", "3"))
DEFAULT_THROTTLE_SECONDS = float(os.getenv("RAG_EMBEDDING_THROTTLE_SECONDS", "0"))


def embedding_model() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_texts(texts: list[str], model_ref: str | None = None) -> list[list[float]]:
    model_ref = model_ref or embedding_model()
    provider, model = parse_model_ref(model_ref)
    clean_texts = [text or "" for text in texts]
    if not clean_texts:
        return []
    if provider == "gemini":
        return _gemini_embeddings(clean_texts, model)
    if provider == "local":
        return [_local_hash_embedding(text) for text in clean_texts]
    if provider in {"openai", "openrouter", "ollama"}:
        return _openai_compatible_embeddings(clean_texts, provider, model)
    raise RuntimeError(f"unsupported_embedding_provider:{provider}")


def embed_text(text: str, model_ref: str | None = None) -> list[float]:
    return embed_texts([text], model_ref=model_ref)[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _gemini_embeddings(texts: list[str], model: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in _batches(texts, DEFAULT_BATCH_SIZE):
        for text in batch:
            response = _with_retry(lambda: GEMINI_CLIENT.models.embed_content(
                model=model,
                contents=text,
            ))
            embeddings = getattr(response, "embeddings", None) or []
            if not embeddings:
                raise RuntimeError("gemini_embedding_empty_response")
            values = getattr(embeddings[0], "values", None)
            if values is None:
                raise RuntimeError("gemini_embedding_missing_values")
            vectors.append([float(value) for value in values])
            _throttle()
    return vectors


def _openai_compatible_embeddings(texts: list[str], provider: str, model: str) -> list[list[float]]:
    base_url, api_key, extra_headers = _openai_compatible_config(provider)
    headers = {"Content-Type": "application/json", **extra_headers}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    vectors: list[list[float]] = []
    for batch in _batches(texts, DEFAULT_BATCH_SIZE):
        payload = {"model": model, "input": batch}
        response = _with_retry(lambda: requests.post(
            f"{base_url.rstrip('/')}/embeddings",
            headers=headers,
            json=payload,
            timeout=60,
        ))
        if not response.ok:
            detail = response.text[:500] if response.text else response.reason
            raise RuntimeError(f"{provider}:{model} embeddings HTTP {response.status_code}: {detail}")
        data = response.json()
        rows = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        if len(rows) != len(batch):
            raise RuntimeError("embedding_count_mismatch")
        for row in rows:
            vectors.append([float(value) for value in row.get("embedding") or []])
        _throttle()
    return vectors


def _local_hash_embedding(text: str) -> list[float]:
    vector = [0.0] * LOCAL_HASH_DIM
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % LOCAL_HASH_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\wğüşöçıİĞÜŞÖÇ]+", (text or "").lower(), flags=re.UNICODE)
    return _expand_tokens(raw_tokens)


def _expand_tokens(tokens: list[str]) -> list[str]:
    synonyms = {
        "ayda": ["aylik"],
        "odemem": ["odeme", "ucret", "fiyat"],
        "odeme": ["ucret", "fiyat"],
        "tutar": ["ucret", "fiyat", "tl"],
        "ucreti": ["ucret", "fiyat", "tl"],
        "ucret": ["fiyat", "tl"],
        "fiyati": ["fiyat", "ucret", "tl"],
        "fiyat": ["ucret", "tl"],
        "kapsamaz": ["dahil", "degildir", "olmayanlar"],
        "kapsar": ["dahil", "kapsam"],
        "olmayanlari": ["olmayanlar", "dahil", "degildir"],
        "yardimi": ["destegi"],
        "denetimi": ["incelemesi"],
        "calismalarini": ["yonetimi"],
        "operatoru": ["operator"],
    }
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(synonyms.get(token, []))
    return expanded


def _batches(items: list[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [items[index:index + size] for index in range(0, len(items), size)]


def _with_retry(fn):
    last_exc = None
    for attempt in range(max(1, DEFAULT_RETRIES)):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= DEFAULT_RETRIES - 1:
                break
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    raise last_exc


def _throttle() -> None:
    if DEFAULT_THROTTLE_SECONDS > 0:
        time.sleep(DEFAULT_THROTTLE_SECONDS)
