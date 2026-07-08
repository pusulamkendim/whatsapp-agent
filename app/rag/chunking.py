from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


DEFAULT_CHUNK_TOKENS = 750
DEFAULT_OVERLAP_TOKENS = 120


@dataclass(frozen=True)
class ChunkSpec:
    chunk_index: int
    title_path: str
    content: str
    content_hash: str
    token_count: int
    metadata: dict


def estimate_tokens(text: str) -> int:
    words = re.findall(r"\S+", text or "")
    return max(1, int(len(words) * 1.25)) if words else 0


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def chunk_document(
    content: str,
    filename: str = "",
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[ChunkSpec]:
    sections = _markdown_sections(content or "", filename)
    chunks: list[ChunkSpec] = []
    for title_path, section_text in sections:
        for piece in _split_section(section_text, max_tokens, overlap_tokens):
            normalized = _normalize_chunk(title_path, piece)
            if not normalized:
                continue
            start_char = content.find(piece[:120].strip()) if piece.strip() else -1
            end_char = start_char + len(piece) if start_char >= 0 else -1
            chunks.append(ChunkSpec(
                chunk_index=len(chunks),
                title_path=title_path,
                content=normalized,
                content_hash=content_hash(normalized),
                token_count=estimate_tokens(normalized),
                metadata={
                    "filename": filename,
                    "start_char": start_char,
                    "end_char": end_char,
                },
            ))
    return chunks


def _markdown_sections(content: str, filename: str) -> list[tuple[str, str]]:
    lines = content.splitlines()
    heading_stack: list[tuple[int, str]] = []
    current_title = filename or "document"
    current_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_title, text))

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            current_lines.append(line)
            continue

        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        current_title = " > ".join([filename or "document", *[item[1] for item in heading_stack]])
        current_lines = []

    flush()
    if sections:
        return sections
    text = content.strip()
    return [(filename or "document", text)] if text else []


def _split_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > max_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current = []
                current_tokens = 0
            pieces.extend(_split_words(paragraph, max_tokens, overlap_tokens))
            continue

        if current and current_tokens + paragraph_tokens > max_tokens:
            pieces.append("\n\n".join(current))
            current = _overlap_paragraphs(current, overlap_tokens)
            current_tokens = estimate_tokens("\n\n".join(current))

        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _split_words(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = re.findall(r"\S+", text or "")
    if not words:
        return []
    approx_words = max(80, int(max_tokens / 1.25))
    overlap_words = min(max(0, int(overlap_tokens / 1.25)), approx_words // 2)
    step = max(1, approx_words - overlap_words)
    pieces = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + approx_words]).strip()
        if piece:
            pieces.append(piece)
        if start + approx_words >= len(words):
            break
    return pieces


def _overlap_paragraphs(paragraphs: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    selected: list[str] = []
    total = 0
    for paragraph in reversed(paragraphs):
        tokens = estimate_tokens(paragraph)
        if selected and total + tokens > overlap_tokens:
            break
        selected.append(paragraph)
        total += tokens
    return list(reversed(selected))


def _normalize_chunk(title_path: str, text: str) -> str:
    body = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not body:
        return ""
    return f"Kaynak basligi: {title_path}\n\n{body}"
