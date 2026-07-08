import base64
import html
import json
import os
import re
import shutil
import textwrap
import uuid
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.config import GEMINI_API_KEY


DATA_DIR = Path(os.getenv("IMAGE_LOCALIZATION_DIR", "data/image_localization"))
MAX_IMAGE_BYTES = int(os.getenv("IMAGE_LOCALIZATION_MAX_BYTES", str(12 * 1024 * 1024)))
SUPPORTED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class StoredUpload:
    filename: str
    content_type: str
    path: str


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def safe_json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def store_upload(filename: str, content_type: str, raw: bytes) -> StoredUpload:
    ensure_storage()
    clean_name = os.path.basename(filename or "image.png")
    extension = Path(clean_name).suffix.lower() or ".png"
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("unsupported_file_type")
    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError("unsupported_content_type")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("file_too_large")
    if not raw:
        raise ValueError("empty_file")

    upload_id = uuid.uuid4().hex
    target = DATA_DIR / f"{upload_id}{extension}"
    target.write_bytes(raw)
    return StoredUpload(filename=clean_name, content_type=content_type or "image/png", path=str(target))


def store_url_image(url: str) -> StoredUpload:
    return store_url_images(url)[0]


def store_url_images(url: str) -> list[StoredUpload]:
    clean_url = (url or "").strip()
    parsed = urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("invalid_url")

    response = _get_url(clean_url, accept_html=_is_instagram_post_url(clean_url))
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type in SUPPORTED_CONTENT_TYPES:
        return [_store_remote_image_response(response, clean_url)]

    image_urls = _extract_instagram_image_urls(response.text, clean_url)
    if not image_urls:
        image_url = _extract_meta_image_url(response.text, clean_url)
        if image_url:
            image_urls = [image_url]
    if not image_urls:
        image_url = _instagram_media_url(clean_url)
        if image_url:
            image_urls = [image_url]
    if not image_urls:
        raise ValueError("image_url_not_found")

    stored = []
    for index, image_url in enumerate(_unique_urls(image_urls), start=1):
        try:
            image_response = _get_url(image_url)
            image_content_type = image_response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if image_content_type not in SUPPORTED_CONTENT_TYPES:
                continue
            stored.append(_store_remote_image_response(image_response, image_url, index=index))
        except Exception as exc:
            print(f"⚠️ Remote görsel atlandı ({index}): {type(exc).__name__}")
    if not stored:
        raise ValueError("remote_images_not_downloaded")
    return stored


def _store_remote_image_response(response: requests.Response, source_url: str, index: int | None = None) -> StoredUpload:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    raw = _bounded_content(response)
    extension = _extension_for_content_type(content_type)
    filename = Path(urlparse(source_url).path).name or f"remote{extension}"
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        filename = f"{Path(filename).stem or 'remote'}{extension}"
    if index is not None:
        filename = f"{Path(filename).stem[:80]}-{index:02d}{Path(filename).suffix.lower() or extension}"
    return store_upload(filename, content_type, raw)


def _get_url(url: str, accept_html: bool = False) -> requests.Response:
    accept = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" if accept_html else (
        "text/html,image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
    )
    user_agent = "Mozilla/5.0" if accept_html else (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    )
    headers = {
        "User-Agent": user_agent,
        "Accept": accept,
    }
    response = requests.get(url, headers=headers, timeout=20, stream=True)
    if not response.ok:
        raise ValueError(f"remote_fetch_failed:{response.status_code}")
    return response


def _bounded_content(response: requests.Response) -> bytes:
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise ValueError("remote_file_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_meta_image_url(markup: str, base_url: str) -> str:
    patterns = [
        r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::secure_url)?["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, markup or "", flags=re.IGNORECASE)
        if match:
            return urljoin(base_url, html.unescape(match.group(1)))
    return ""


def _extract_instagram_image_urls(markup: str, base_url: str) -> list[str]:
    if not _is_instagram_post_url(base_url):
        return []

    urls = []
    media = _extract_json_object_after_key(markup or "", '"shortcode_media":')
    if media:
        urls.extend(_instagram_urls_from_media_item(media))

    sidecar = _extract_json_object_after_key(markup or "", '"edge_sidecar_to_children":')
    if sidecar:
        urls.extend(_instagram_urls_from_media_item(sidecar))

    for key in ('"edge_sidecar_to_children":', '"carousel_media":'):
        media = _extract_json_array_after_key(markup or "", key)
        for item in media:
            urls.extend(_instagram_urls_from_media_item(item))

    if not urls:
        image_versions = _extract_json_object_after_key(markup or "", '"image_versions2":')
        best = _best_instagram_candidate_url(image_versions)
        if best:
            urls.append(best)

    return _unique_urls(urls)


def _extract_json_object_after_key(markup: str, key: str) -> Any:
    start = markup.find(key)
    if start < 0:
        return None
    object_start = markup.find("{", start + len(key))
    if object_start < 0:
        return None

    in_string = False
    escaped = False
    depth = 0
    for pos in range(object_start, len(markup)):
        char = markup[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(markup[object_start:pos + 1])
                    return value
                except ValueError:
                    return None
    return None


def _collect_instagram_image_urls(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"display_url", "display_uri", "thumbnail_src", "src"} and isinstance(item, str):
                urls.append(html.unescape(item))
            elif key == "url" and isinstance(item, str):
                urls.append(html.unescape(item))
            else:
                urls.extend(_collect_instagram_image_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_collect_instagram_image_urls(item))
    return urls


def _extract_json_array_after_key(markup: str, key: str) -> list[Any]:
    start = markup.find(key)
    if start < 0:
        return []
    array_start = markup.find("[", start + len(key))
    if array_start < 0:
        return []

    in_string = False
    escaped = False
    depth = 0
    for pos in range(array_start, len(markup)):
        char = markup[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(markup[array_start:pos + 1])
                    return value if isinstance(value, list) else []
                except ValueError:
                    return []
    return []


def _instagram_urls_from_media_item(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return []

    node = item.get("node")
    if isinstance(node, dict):
        return _instagram_urls_from_media_item(node)

    for children_key in ("edge_sidecar_to_children", "carousel_media"):
        children = item.get(children_key)
        if isinstance(children, dict):
            urls = _instagram_urls_from_media_item(children)
            if urls:
                return urls
        if isinstance(children, list):
            urls = []
            for child in children:
                urls.extend(_instagram_urls_from_media_item(child))
            if urls:
                return urls

    edges = item.get("edges")
    if isinstance(edges, list):
        urls = []
        for edge in edges:
            urls.extend(_instagram_urls_from_media_item(edge))
        if urls:
            return urls

    best_candidate = _best_instagram_candidate_url(item.get("image_versions2"))
    if best_candidate:
        return [best_candidate]

    display_uri = item.get("display_uri") or item.get("display_url")
    if isinstance(display_uri, str):
        return [html.unescape(display_uri)]

    if isinstance(item.get("url"), str):
        return [html.unescape(item["url"])]

    return []


def _best_instagram_candidate_url(image_versions: Any) -> str:
    if not isinstance(image_versions, dict):
        return ""
    candidates = image_versions.get("candidates")
    if not isinstance(candidates, list):
        return ""

    urls = []
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("url"), str):
            urls.append(html.unescape(candidate["url"]))
    if not urls:
        return ""

    uncropped = [url for url in urls if not _instagram_url_has_crop(url)]
    return (uncropped or urls)[0]


def _instagram_url_has_crop(url: str) -> bool:
    match = re.search(r"[?&]stp=([^&]+)", url)
    if not match:
        return False
    stp = match.group(1)
    return bool(re.match(r"c\d+[._]", stp)) or "_s" in stp and "x" in stp and "_p" not in stp


def _unique_urls(urls: list[str]) -> list[str]:
    seen = set()
    unique = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(url)
    return unique


def _instagram_media_url(url: str) -> str:
    if not _is_instagram_post_url(url):
        return ""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] in {"p", "reel", "tv"}:
        return f"https://www.instagram.com/{parts[0]}/{parts[1]}/media/?size=l"
    return ""


def _is_instagram_post_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {"instagram.com", "www.instagram.com"}:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    return len(parts) >= 2 and parts[0] in {"p", "reel", "tv"}


def _extension_for_content_type(content_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }.get(content_type, ".png")


def prepare_asset(original_path: str) -> dict:
    image = Image.open(original_path).convert("RGB")
    cropped_path = original_path
    crop = {"x": 0, "y": 0, "width": image.width, "height": image.height, "mode": "original"}
    return {
        "cropped_path": cropped_path,
        "crop": crop,
        "ocr": [],
        "translations": [],
    }


SUPPORTED_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
}


ProgressCallback = Callable[[str, str, dict | None], None]


def _emit_progress(callback: ProgressCallback | None, step: str, message: str, meta: dict | None = None) -> None:
    if callback:
        callback(step, message, meta or {})


def run_ocr_pipeline(image_path: str, options: dict | None = None, progress_callback: ProgressCallback | None = None) -> dict:
    options = options or {}
    engine = _selected_ocr_engine(options)
    _emit_progress(progress_callback, "prepare", f"OCR hazırlandı. Motor: {engine}", {"engine": engine})
    image = Image.open(image_path).convert("RGB")
    _emit_progress(progress_callback, "detect", "Metin bölgeleri aranıyor.", {"engine": engine})
    ocr_items = extract_text_boxes(image_path, options=options, progress_callback=progress_callback)
    _emit_progress(progress_callback, "detected", f"{len(ocr_items)} metin kutusu bulundu.", {"count": len(ocr_items)})
    _emit_progress(
        progress_callback,
        "translate",
        f"Çeviri başlıyor. Model: {options.get('translation_model') or os.getenv('IMAGE_TRANSLATION_MODEL') or 'gemini-2.5-flash'}",
        {"model": options.get("translation_model") or os.getenv("IMAGE_TRANSLATION_MODEL") or "gemini-2.5-flash"},
    )
    translations = translate_text_boxes(ocr_items, model=options.get("translation_model"))
    _emit_progress(progress_callback, "fit", "Kutular çeviri metnine göre ayarlanıyor.", {"count": len(translations)})
    translations = fit_text_boxes_to_content(translations, image.size)
    _emit_progress(progress_callback, "done", "OCR ve çeviri tamamlandı.", {"count": len(translations)})
    return {
        "ocr": ocr_items,
        "translations": translations,
    }


def extract_text_boxes(image_path: str, options: dict | None = None, progress_callback: ProgressCallback | None = None) -> list[dict]:
    options = options or {}
    engine = _selected_ocr_engine(options)
    local_enabled = _local_ocr_enabled()
    use_local = local_enabled and engine in {"auto", "local", "easyocr", "paddleocr"}
    use_gemini = engine in {"auto", "gemini"}
    best_local: list[dict] = []

    if engine in {"local", "easyocr", "paddleocr"} and not local_enabled:
        _emit_progress(progress_callback, "local-ocr", "Yerel OCR bu ortamda kapalı; Gemini Vision kullanılacak.", {"engine": engine})
        use_gemini = True

    if use_local:
        extractors = []
        if engine in {"auto", "local", "paddleocr"}:
            extractors.append(_extract_with_paddleocr)
        if engine in {"auto", "local", "easyocr"}:
            extractors.append(_extract_with_easyocr)

        for extractor in extractors:
            try:
                extractor_name = extractor.__name__.replace("_extract_with_", "")
                _emit_progress(progress_callback, "local-ocr", f"{extractor_name} çalışıyor.", {"engine": extractor_name})
                items = extractor(image_path)
                if items:
                    normalized = normalize_ocr_items(items)
                    _emit_progress(
                        progress_callback,
                        "local-ocr",
                        f"{extractor_name} {len(normalized)} metin adayı buldu.",
                        {"engine": extractor_name, "count": len(normalized), "quality": round(_ocr_quality_score(normalized), 3)},
                    )
                    if _ocr_result_is_reliable(normalized):
                        _emit_progress(progress_callback, "local-ocr", f"{extractor_name} sonucu yeterli bulundu.", {"engine": extractor_name})
                        return normalized
                    if _ocr_quality_score(normalized) > _ocr_quality_score(best_local):
                        best_local = normalized
                else:
                    _emit_progress(progress_callback, "local-ocr", f"{extractor_name} metin bulamadı.", {"engine": extractor_name, "count": 0})
            except Exception as exc:
                _emit_progress(progress_callback, "local-ocr", f"{extractor.__name__} atlandı: {type(exc).__name__}", {"error": type(exc).__name__})
                print(f"⚠️ OCR atlandı ({extractor.__name__}): {type(exc).__name__}")

    if use_gemini and GEMINI_API_KEY:
        try:
            selected_model = _safe_gemini_model(options.get("ocr_model") or os.getenv("IMAGE_OCR_MODEL") or "gemini-2.5-flash")
            _emit_progress(progress_callback, "gemini-ocr", f"Gemini Vision OCR çalışıyor. Model: {selected_model}", {"model": selected_model})
            items = _extract_with_gemini_vision(image_path, model=options.get("ocr_model"))
            normalized = normalize_ocr_items(items)
            _emit_progress(progress_callback, "gemini-ocr", f"Gemini {len(normalized)} metin adayı buldu.", {"count": len(normalized), "model": selected_model})
            merged = _merge_text_with_local_geometry(best_local, normalized)
            if merged:
                _emit_progress(progress_callback, "merge", "Gemini metni yerel OCR geometrisiyle birleştirildi.", {"count": len(merged)})
                return merged
            if normalized and _ocr_quality_score(normalized) >= _ocr_quality_score(best_local):
                _emit_progress(progress_callback, "gemini-ocr", "Gemini sonucu seçildi.", {"model": selected_model})
                return normalized
        except Exception as exc:
            _emit_progress(progress_callback, "gemini-ocr", f"Gemini OCR atlandı: {type(exc).__name__}", {"error": type(exc).__name__})
            print(f"⚠️ OCR atlandı (_extract_with_gemini_vision): {type(exc).__name__}")

    if best_local:
        _emit_progress(progress_callback, "local-ocr", "En iyi yerel OCR sonucu seçildi.", {"count": len(best_local)})
        return best_local
    _emit_progress(progress_callback, "detect", "Metin kutusu bulunamadı.", {"count": 0})
    return []


def _selected_ocr_engine(options: dict | None = None) -> str:
    options = options or {}
    engine = (options.get("engine") or os.getenv("IMAGE_OCR_ENGINE") or "auto").lower()
    if engine not in {"auto", "gemini", "local", "easyocr", "paddleocr"}:
        engine = "auto"
    if not _local_ocr_enabled() and engine == "auto":
        return "gemini"
    return engine


def _local_ocr_enabled() -> bool:
    override = os.getenv("IMAGE_LOCAL_OCR_ENABLE")
    if override is not None:
        return override.lower() in {"1", "true", "yes", "on"}
    return not bool(os.getenv("COOLIFY_RESOURCE_UUID") or os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT"))


def _ocr_result_is_reliable(items: list[dict]) -> bool:
    if not items:
        return False
    return _ocr_quality_score(items) >= 0.82 and not _has_overlapping_duplicate_text(items)


def _ocr_quality_score(items: list[dict]) -> float:
    if not items:
        return 0.0
    confidences = [float(item.get("confidence") or 0) for item in items]
    score = sum(confidences) / len(confidences)
    if min(confidences) < 0.7:
        score -= 0.16
    if _has_overlapping_duplicate_text(items):
        score -= 0.24
    return max(0.0, min(1.0, score))


def _has_overlapping_duplicate_text(items: list[dict]) -> bool:
    for index, first in enumerate(items):
        first_text = clean_text(first.get("source_text") or "").lower()
        if not first_text:
            continue
        first_box = _ocr_item_rect(first)
        for second in items[index + 1:]:
            second_text = clean_text(second.get("source_text") or "").lower()
            if not second_text:
                continue
            shorter, longer = sorted((first_text, second_text), key=len)
            if len(shorter) < 3 or shorter not in longer:
                continue
            if _rect_overlap_ratio(first_box, _ocr_item_rect(second)) > 0.35:
                return True
    return False


def _ocr_item_rect(item: dict) -> tuple[float, float, float, float]:
    x = float(item.get("x") or 0)
    y = float(item.get("y") or 0)
    return (x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0))


def _rect_overlap_ratio(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return overlap / min(first_area, second_area)


def _merge_text_with_local_geometry(local_items: list[dict], text_items: list[dict]) -> list[dict]:
    if not local_items or not text_items:
        return []
    local_items = filter_ocr_noise_items(local_items)
    text_items = filter_ocr_noise_items(text_items)
    if not local_items or not text_items:
        return []

    if len(text_items) == 1:
        if len(local_items) > 1 and should_keep_local_geometry_for_single_gemini(local_items, text_items[0]):
            return local_items
        text = clean_text(text_items[0].get("source_text") or "")
        if not text:
            return []
        rects = [_ocr_item_rect(item) for item in local_items]
        x1 = min(rect[0] for rect in rects)
        y1 = min(rect[1] for rect in rects)
        x2 = max(rect[2] for rect in rects)
        y2 = max(rect[3] for rect in rects)
        return [{
            "id": "text_1",
            "source_text": text,
            "translated_text": "",
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "confidence": max(float(text_items[0].get("confidence") or 0), _ocr_quality_score(local_items)),
            "align": "center",
        }]

    if len(local_items) != len(text_items):
        return []

    merged = []
    for index, (local, text_item) in enumerate(zip(local_items, text_items), start=1):
        copy = dict(local)
        copy["id"] = local.get("id") or f"text_{index}"
        copy["source_text"] = clean_text(text_item.get("source_text") or local.get("source_text") or "")
        copy["confidence"] = max(float(local.get("confidence") or 0), float(text_item.get("confidence") or 0))
        merged.append(copy)
    return merged


def should_keep_local_geometry_for_single_gemini(local_items: list[dict], text_item: dict) -> bool:
    text_rect = _ocr_item_rect(text_item)
    text_area = max(1.0, (text_rect[2] - text_rect[0]) * (text_rect[3] - text_rect[1]))
    local_rects = [_ocr_item_rect(item) for item in local_items]
    local_areas = [max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1])) for rect in local_rects]
    union = (
        min(rect[0] for rect in local_rects),
        min(rect[1] for rect in local_rects),
        max(rect[2] for rect in local_rects),
        max(rect[3] for rect in local_rects),
    )
    union_area = max(1.0, (union[2] - union[0]) * (union[3] - union[1]))
    fill_ratio = sum(local_areas) / union_area
    text = clean_text(text_item.get("source_text") or "").lower()
    local_text = " ".join(clean_text(item.get("source_text") or "").lower() for item in local_items)
    local_words = {word for word in re.findall(r"[a-zA-Z]{3,}", local_text)}
    text_words = {word for word in re.findall(r"[a-zA-Z]{3,}", text)}
    overlap = len(local_words & text_words) / max(1, len(local_words))
    return (text_area > union_area * 1.55 or fill_ratio < 0.42) and overlap >= 0.45


def _extract_with_paddleocr(image_path: str) -> list[dict]:
    ocr = get_paddleocr_reader()
    result = ocr.ocr(image_path, cls=True)
    items = []
    for page in result or []:
        for line in page or []:
            box = line[0]
            text, confidence = line[1]
            items.append({"box": box, "text": text, "confidence": float(confidence)})
    return items


def _extract_with_easyocr(image_path: str) -> list[dict]:
    reader = get_easyocr_reader()
    result = reader.readtext(
        image_path,
        detail=1,
        paragraph=False,
        decoder=os.getenv("EASYOCR_DECODER", "beamsearch"),
        beamWidth=int(os.getenv("EASYOCR_BEAM_WIDTH", "8")),
        contrast_ths=float(os.getenv("EASYOCR_CONTRAST_THS", "0.05")),
        adjust_contrast=float(os.getenv("EASYOCR_ADJUST_CONTRAST", "0.7")),
        text_threshold=float(os.getenv("EASYOCR_TEXT_THRESHOLD", "0.55")),
        low_text=float(os.getenv("EASYOCR_LOW_TEXT", "0.3")),
        link_threshold=float(os.getenv("EASYOCR_LINK_THRESHOLD", "0.3")),
        mag_ratio=float(os.getenv("EASYOCR_MAG_RATIO", "1.5")),
    )
    return [{"box": box, "text": text, "confidence": float(confidence)} for box, text, confidence in result]


@lru_cache(maxsize=1)
def get_easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


@lru_cache(maxsize=1)
def get_paddleocr_reader():
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def warm_ocr_models() -> dict[str, str]:
    result: dict[str, str] = {}
    if os.getenv("OCR_WARMUP_ENABLE", "true").lower() in {"0", "false", "no"}:
        return {"status": "skipped"}

    for name, loader in (("paddleocr", get_paddleocr_reader), ("easyocr", get_easyocr_reader)):
        try:
            loader()
            result[name] = "ready"
        except Exception as exc:
            result[name] = f"error:{type(exc).__name__}"
            print(f"⚠️ OCR warmup atlandı ({name}): {type(exc).__name__}")
    return result


def _extract_with_gemini_vision(image_path: str, model: str | None = None) -> list[dict]:
    if not GEMINI_API_KEY:
        return []
    from app.llm import GEMINI_CLIENT, record_gemini_usage
    from google.genai import types

    image = Image.open(image_path)
    mime_type = "image/png"
    if image_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith(".webp"):
        mime_type = "image/webp"

    prompt = (
        "Detect every distinct text block in this image for image localization. "
        "Return only JSON array. Each item must be: "
        "{\"text\":\"...\",\"x\":0,\"y\":0,\"width\":0,\"height\":0,\"confidence\":0.0}. "
        f"Coordinates must be pixel coordinates in the image size {image.width}x{image.height}. "
        "Keep separate text blocks separate. Merge only lines that belong to the same paragraph or quote. "
        "Do not include decorative icons, usernames, or non-text image details."
    )
    selected_model = _safe_gemini_model(model or os.getenv("IMAGE_OCR_MODEL") or "gemini-2.5-flash")
    response = GEMINI_CLIENT.models.generate_content(
        model=selected_model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=0.0),
    )
    record_gemini_usage(f"gemini:{selected_model}", response, source="image-localizer", operation="ocr")
    raw = response.candidates[0].content.parts[0].text.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    values = json.loads(raw)
    if not isinstance(values, list):
        return []

    items = []
    for value in values:
        if not isinstance(value, dict):
            continue
        text = clean_text(value.get("text") or "")
        if not text:
            continue
        x = int(float(value.get("x") or 0))
        y = int(float(value.get("y") or 0))
        w = int(float(value.get("width") or 0))
        h = int(float(value.get("height") or 0))
        if w <= 4 or h <= 4:
            continue
        x = max(0, min(x, image.width - 1))
        y = max(0, min(y, image.height - 1))
        x2 = min(image.width, x + w)
        y2 = min(image.height, y + h)
        items.append({
            "box": [[x, y], [x2, y], [x2, y2], [x, y2]],
            "text": text,
            "confidence": float(value.get("confidence") or 0.85),
        })
    return items


def normalize_ocr_items(items: list[dict]) -> list[dict]:
    normalized = []
    for index, item in enumerate(items, start=1):
        box = item.get("box") or []
        xs = [point[0] for point in box if len(point) >= 2]
        ys = [point[1] for point in box if len(point) >= 2]
        text = clean_text(item.get("text") or "")
        if not xs or not ys or not text:
            continue
        normalized.append({
            "id": f"text_{index}",
            "source_text": text,
            "translated_text": "",
            "x": int(min(xs)),
            "y": int(min(ys)),
            "width": int(max(xs) - min(xs)),
            "height": int(max(ys) - min(ys)),
            "confidence": round(float(item.get("confidence") or 0), 4),
            "align": "center",
        })
    return merge_nearby_text_lines(filter_ocr_noise_items(normalized))


def filter_ocr_noise_items(items: list[dict]) -> list[dict]:
    filtered = []
    for item in items:
        text = clean_text(item.get("source_text") or item.get("text") or "")
        confidence = float(item.get("confidence") or 0)
        alpha_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text))
        looks_like_handle = "@" in text or text.startswith("#")
        if looks_like_handle:
            continue
        if confidence < 0.45 and (looks_like_handle or alpha_count < 4):
            continue
        if confidence < 0.35:
            continue
        filtered.append(item)
    return filtered


def merge_nearby_text_lines(items: list[dict]) -> list[dict]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: (item["y"], item["x"]))
    if should_merge_as_quote_block(ordered):
        return [merge_text_group(ordered, 1)]

    groups: list[list[dict]] = []
    for item in ordered:
        placed = False
        for group in groups:
            last = group[-1]
            same_row = abs((item["y"] + item["height"] / 2) - (last["y"] + last["height"] / 2)) < max(item["height"], last["height"]) * 0.65
            horizontal_gap = _horizontal_gap(item, last)
            same_column = abs((item["x"] + item["width"] / 2) - (last["x"] + last["width"] / 2)) < max(item["width"], last["width"]) * 0.8
            vertical_gap = item["y"] - (last["y"] + last["height"])
            close_y = -max(8, last["height"] * 0.35) <= vertical_gap < max(28, last["height"] * 1.6)
            if (same_row and horizontal_gap <= max(36, last["height"] * 1.4)) or (same_column and close_y):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    merged = [merge_text_group(group, index) for index, group in enumerate(groups, start=1)]
    return merge_quote_like_groups(merged)


def should_merge_as_quote_block(items: list[dict]) -> bool:
    if len(items) < 2 or len(items) > 5:
        return False
    ordered = sorted(items, key=lambda item: (item["y"], item["x"]))
    median_height = float(np.median([item["height"] for item in ordered]))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            same_row = abs((first["y"] + first["height"] / 2) - (second["y"] + second["height"] / 2)) < max(first["height"], second["height"]) * 0.75
            if not same_row:
                continue
            left, right = sorted((first, second), key=lambda item: item["x"])
            gap = right["x"] - (left["x"] + left["width"])
            if gap > max(90, median_height * 2.5):
                return False
    for previous, current in zip(ordered, ordered[1:]):
        gap = current["x"] - (previous["x"] + previous["width"])
        overlap = _horizontal_overlap_ratio(previous, current)
        if gap > max(90, median_height * 2.5) and overlap < 0.12:
            return False
    xs = [item["x"] for item in items]
    x2s = [item["x"] + item["width"] for item in items]
    ys = [item["y"] for item in items]
    y2s = [item["y"] + item["height"] for item in items]
    union_width = max(x2s) - min(xs)
    union_height = max(y2s) - min(ys)
    if union_width <= 0 or union_height <= 0:
        return False
    vertical_span_ok = union_height < max(220, np.median([item["height"] for item in items]) * 4.6)
    horizontal_overlap = any(_horizontal_overlap_ratio(items[i], items[i + 1]) > 0.18 for i in range(len(items) - 1))
    close_stack = all(items[i + 1]["y"] - (items[i]["y"] + items[i]["height"]) < max(34, items[i]["height"] * 1.5) for i in range(len(items) - 1))
    total_chars = sum(len(clean_text(item.get("source_text") or "")) for item in items)
    return vertical_span_ok and (horizontal_overlap or close_stack) and total_chars >= 18


def merge_quote_like_groups(items: list[dict]) -> list[dict]:
    if should_merge_as_quote_block(sorted(items, key=lambda item: (item["y"], item["x"]))):
        return [merge_text_group(items, 1)]
    return items


def merge_text_group(group: list[dict], index: int) -> dict:
    ordered = sorted(group, key=lambda item: (item["y"], item["x"]))
    xs = [item["x"] for item in ordered]
    ys = [item["y"] for item in ordered]
    x2s = [item["x"] + item["width"] for item in ordered]
    y2s = [item["y"] + item["height"] for item in ordered]
    return {
        "id": f"text_{index}",
        "source_text": clean_text(" ".join(item["source_text"] for item in ordered)),
        "translated_text": "",
        "x": int(min(xs)),
        "y": int(min(ys)),
        "width": int(max(x2s) - min(xs)),
        "height": int(max(y2s) - min(ys)),
        "confidence": min(float(item.get("confidence") or 0) for item in ordered),
        "align": "left" if len(ordered) > 1 else ordered[0].get("align", "center"),
    }


def _horizontal_overlap_ratio(first: dict, second: dict) -> float:
    left = max(first["x"], second["x"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    if right <= left:
        return 0.0
    return (right - left) / max(1, min(first["width"], second["width"]))


def _horizontal_gap(first: dict, second: dict) -> float:
    left, right = sorted((first, second), key=lambda item: item["x"])
    return max(0.0, float(right["x"] - (left["x"] + left["width"])))


def translate_text_boxes(items: list[dict], model: str | None = None) -> list[dict]:
    if not items:
        return []
    translated = _translate_with_gemini([item["source_text"] for item in items], model=model)
    result = []
    for index, item in enumerate(items):
        copy = dict(item)
        copy["translated_text"] = translated[index] if index < len(translated) else item["source_text"]
        copy["auto_fit"] = True
        result.append(copy)
    return result


def fit_text_boxes_to_content(items: list[dict], image_size: tuple[int, int], auto_only: bool = True) -> list[dict]:
    if not items:
        return []
    fitted = []
    for item in items:
        copy = dict(item)
        if auto_only and not bool(copy.get("auto_fit", True)):
            fitted.append(copy)
            continue
        fitted.append(fit_single_text_box(copy, image_size))
    return fitted


def fit_single_text_box(item: dict, image_size: tuple[int, int]) -> dict:
    image_width, image_height = image_size
    text = clean_text(item.get("translated_text") or item.get("source_text") or "")
    if not text:
        return item

    width = max(24, min(image_width - 2, int(float(item.get("width") or 220))))
    height = max(20, min(image_height - 2, int(float(item.get("height") or 60))))
    if len(text) > 42 and bool(item.get("auto_fit", True)):
        x = int(float(item.get("x") or 0))
        y = int(float(item.get("y") or 0))
        target_width = min(image_width - x - 2, max(width, int(image_width * 0.72)))
        target_height = min(image_height - y - 2, max(height, int(height * 1.22), 76))
        item["x"] = max(0, min(x, image_width - 2))
        item["y"] = max(0, min(y, image_height - 2))
        width = max(width, target_width)
        height = max(height, target_height)
        item["align"] = item.get("align") or "left"
    font_size = max(10, min(60, estimate_font_size_for_box(text, (width, height))))
    item["width"] = width
    item["height"] = height
    item["font_size_hint"] = font_size
    return item


def estimate_font_size_for_box(text: str, box_size: tuple[int, int]) -> int:
    width, height = box_size
    draw = ImageDraw.Draw(Image.new("RGB", (width, height), "white"))
    preferred_max_lines = 2 if len(text) >= 48 and width >= height * 3.2 else None
    if preferred_max_lines:
        for font_size in range(min(54, max(14, height)), 9, -1):
            font = load_font(font_size)
            lines = wrap_text_for_width(draw, text, font, max(20, width - 8))
            if len(lines) > preferred_max_lines:
                continue
            line_height = max(14, int(font_size * 1.18))
            total_height = line_height * max(1, len(lines))
            widest = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0)
            if total_height <= height - 8 and widest <= width - 8:
                return font_size
    for font_size in range(min(54, max(14, height)), 9, -1):
        font = load_font(font_size)
        lines = wrap_text_for_width(draw, text, font, max(20, width - 8))
        line_height = max(14, int(font_size * 1.18))
        total_height = line_height * max(1, len(lines))
        widest = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0)
        if total_height <= height - 8 and widest <= width - 8:
            return font_size
    return 10


def _translate_with_gemini(texts: list[str], model: str | None = None) -> list[str]:
    if not GEMINI_API_KEY:
        return [local_translation_fallback(text) for text in texts]
    try:
        from app.llm import GEMINI_CLIENT, record_gemini_usage
        from google.genai import types

        prompt = (
            "Translate these short social media image texts into natural Turkish. "
            "Keep the meaning concise, preserve question marks, and return only JSON array of strings.\n\n"
            f"{json.dumps(texts, ensure_ascii=False)}"
        )
        selected_model = _safe_gemini_model(model or os.getenv("IMAGE_TRANSLATION_MODEL") or "gemini-2.5-flash")
        response = GEMINI_CLIENT.models.generate_content(
            model=selected_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        record_gemini_usage(f"gemini:{selected_model}", response, source="image-localizer", operation="translation")
        raw = response.candidates[0].content.parts[0].text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        values = json.loads(raw)
        if isinstance(values, list) and all(isinstance(item, str) for item in values):
            return values
    except Exception as exc:
        print(f"⚠️ Gemini çeviri atlandı: {type(exc).__name__}")
    return [local_translation_fallback(text) for text in texts]


def _safe_gemini_model(model: str) -> str:
    value = (model or "").strip()
    return value if value in SUPPORTED_GEMINI_MODELS else "gemini-2.5-flash"


def local_translation_fallback(text: str) -> str:
    known = {
        "how do i stop my anger, overthinking, jealousy, and negative thoughts?": "Öfkemi, aşırı düşünmeyi, kıskançlığımı ve olumsuz düşüncelerimi nasıl durdurabilirim?",
        "who told you they are the real problem?": "Bunların asıl sorun olduğunu sana kim söyledi?",
    }
    key = clean_text(text).lower()
    return known.get(key, text)


def generate_output(cropped_path: str, approved_texts: list[dict]) -> str:
    ensure_storage()
    image = Image.open(cropped_path).convert("RGB")
    for item in approved_texts:
        text = clean_text(item.get("translated_text") or item.get("source_text") or "")
        if not text:
            continue
        box = text_box(item, image.size)
        pad = max(8, int(min(box[2] - box[0], box[3] - box[1]) * 0.12))
        clear_box = (
            max(0, box[0] - pad),
            max(0, box[1] - pad),
            min(image.width, box[2] + pad),
            min(image.height, box[3] + pad),
        )
        style = estimate_text_style(image, box, clear_box)
        clear_text_region(image, clear_box, style)
        draw = ImageDraw.Draw(image)
        render_fitted_text(
            draw,
            text,
            box,
            align=item.get("align") or "center",
            fill=style["text"],
            max_font_size=max(12, int(float(item.get("font_size_hint") or style["font_size"]))),
            bold=style["bold"],
            outline=style["outline"],
            shadow=style["shadow"],
            font_family=style["font_family"],
        )

    output_path = str(DATA_DIR / f"{Path(cropped_path).stem}_localized_{uuid.uuid4().hex[:8]}.png")
    image.save(output_path, "PNG")
    return output_path


def text_box(item: dict, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    x = int(float(item.get("x") or 0))
    y = int(float(item.get("y") or 0))
    w = int(float(item.get("width") or width * 0.35))
    h = int(float(item.get("height") or height * 0.12))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return (x, y, min(width, max(x + 20, x + w)), min(height, max(y + 20, y + h)))


def estimate_text_style(image: Image.Image, box: tuple[int, int, int, int], clear_box: tuple[int, int, int, int]) -> dict:
    background = estimate_background_color(image, clear_box)
    text_color = estimate_text_color(image, box, background)
    font_size = estimate_font_size(image, box, background)
    outline, shadow = estimate_text_effects(text_color, background, font_size)
    font_family = estimate_font_family(image, box, background)
    bold = estimate_font_bold(image, box, background)
    return {
        "background": background,
        "text": text_color,
        "font_size": font_size,
        "bold": bold,
        "outline": outline,
        "shadow": shadow,
        "font_family": font_family,
    }


def estimate_background_color(image: Image.Image, clear_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    model = estimate_background_gradient(image, clear_box)
    if model is not None:
        color = np.median(model.reshape(-1, 3), axis=0)
        return tuple(int(value) for value in color)

    return estimate_flat_background_color(image, clear_box)


def estimate_flat_background_color(image: Image.Image, clear_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    arr = np.asarray(image)
    x1, y1, x2, y2 = clear_box
    samples = []
    margin = 4
    if y1 > 0:
        samples.append(arr[max(0, y1 - margin):y1, x1:x2])
    if y2 < image.height:
        samples.append(arr[y2:min(image.height, y2 + margin), x1:x2])
    if x1 > 0:
        samples.append(arr[y1:y2, max(0, x1 - margin):x1])
    if x2 < image.width:
        samples.append(arr[y1:y2, x2:min(image.width, x2 + margin)])

    crop = arr[y1:y2, x1:x2]
    if crop.size:
        brightness = crop.mean(axis=2)
        light_threshold = np.percentile(brightness, 65)
        samples.append(crop[brightness >= light_threshold])

    pixels = [sample.reshape(-1, 3) for sample in samples if getattr(sample, "size", 0)]
    if not pixels:
        return (255, 255, 255)
    merged = np.vstack(pixels)
    color = np.median(merged, axis=0)
    return tuple(int(value) for value in color)


def estimate_background_gradient(image: Image.Image, clear_box: tuple[int, int, int, int]) -> np.ndarray | None:
    arr = np.asarray(image).astype(np.float32)
    x1, y1, x2, y2 = clear_box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    sample_width = max(3, min(12, int(min(width, height) * 0.18)))
    coords: list[np.ndarray] = []
    colors: list[np.ndarray] = []

    def add_sample(region: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
        if not region.size:
            return
        pixels = region.reshape(-1, 3)
        points = np.column_stack((xs.reshape(-1), ys.reshape(-1)))
        if len(pixels) > 6000:
            step = max(1, len(pixels) // 6000)
            pixels = pixels[::step]
            points = points[::step]
        coords.append(points.astype(np.float32))
        colors.append(pixels.astype(np.float32))

    if y1 > 0:
        sy1 = max(0, y1 - sample_width)
        region = arr[sy1:y1, x1:x2]
        yy, xx = np.mgrid[sy1:y1, x1:x2]
        add_sample(region, xx, yy)
    if y2 < image.height:
        sy2 = min(image.height, y2 + sample_width)
        region = arr[y2:sy2, x1:x2]
        yy, xx = np.mgrid[y2:sy2, x1:x2]
        add_sample(region, xx, yy)
    if x1 > 0:
        sx1 = max(0, x1 - sample_width)
        region = arr[y1:y2, sx1:x1]
        yy, xx = np.mgrid[y1:y2, sx1:x1]
        add_sample(region, xx, yy)
    if x2 < image.width:
        sx2 = min(image.width, x2 + sample_width)
        region = arr[y1:y2, x2:sx2]
        yy, xx = np.mgrid[y1:y2, x2:sx2]
        add_sample(region, xx, yy)

    if not coords:
        return None

    points = np.vstack(coords)
    values = np.vstack(colors)
    if len(values) < 24:
        return None

    # Remove high-contrast outliers from text/artifacts along the border before fitting.
    median = np.median(values, axis=0)
    distance = np.linalg.norm(values - median, axis=1)
    keep = distance <= np.percentile(distance, 82)
    points = points[keep]
    values = values[keep]
    if len(values) < 24:
        return None

    nx = points[:, 0] / max(1, image.width - 1)
    ny = points[:, 1] / max(1, image.height - 1)
    design = np.column_stack((nx, ny, np.ones_like(nx)))
    coeffs = []
    for channel in range(3):
        coef, *_ = np.linalg.lstsq(design, values[:, channel], rcond=None)
        coeffs.append(coef)

    yy, xx = np.mgrid[y1:y2, x1:x2]
    gx = xx.astype(np.float32) / max(1, image.width - 1)
    gy = yy.astype(np.float32) / max(1, image.height - 1)
    patch = np.zeros((height, width, 3), dtype=np.float32)
    for channel, coef in enumerate(coeffs):
        patch[:, :, channel] = coef[0] * gx + coef[1] * gy + coef[2]
    return np.clip(patch, 0, 255).astype(np.uint8)


def estimate_text_color(image: Image.Image, box: tuple[int, int, int, int], background: tuple[int, int, int]) -> tuple[int, int, int]:
    arr = np.asarray(image)
    x1, y1, x2, y2 = box
    crop = arr[y1:y2, x1:x2]
    if not crop.size:
        return (20, 20, 20)
    bg = np.array(background)
    distance = np.linalg.norm(crop.astype(float) - bg, axis=2)
    threshold = max(28, float(np.percentile(distance, 88)))
    text_pixels = crop[distance >= threshold]
    if text_pixels.size < 9:
        brightness = crop.mean(axis=2)
        text_pixels = crop[brightness <= np.percentile(brightness, 18)]
    if text_pixels.size < 9:
        return (20, 20, 20) if sum(background) > 384 else (245, 245, 245)
    color = np.median(text_pixels.reshape(-1, 3), axis=0)
    return tuple(int(value) for value in color)


def estimate_font_size(image: Image.Image, box: tuple[int, int, int, int], background: tuple[int, int, int]) -> int:
    arr = np.asarray(image)
    x1, y1, x2, y2 = box
    crop = arr[y1:y2, x1:x2]
    if not crop.size:
        return min(46, max(14, y2 - y1))
    bg = np.array(background)
    distance = np.linalg.norm(crop.astype(float) - bg, axis=2)
    mask = distance >= max(24, np.percentile(distance, 82))
    row_has_text = mask.mean(axis=1) > 0.025
    runs = []
    start = None
    for index, active in enumerate(row_has_text):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= 3:
                runs.append(index - start)
            start = None
    if start is not None and len(row_has_text) - start >= 3:
        runs.append(len(row_has_text) - start)
    if not runs:
        return min(46, max(14, int((y2 - y1) * 0.45)))
    return min(54, max(12, int(np.median(runs) * 1.4)))


def estimate_font_family(image: Image.Image, box: tuple[int, int, int, int], background: tuple[int, int, int]) -> str:
    arr = np.asarray(image)
    x1, y1, x2, y2 = box
    crop = arr[y1:y2, x1:x2]
    if not crop.size:
        return "humanist"
    bg = np.array(background)
    distance = np.linalg.norm(crop.astype(float) - bg, axis=2)
    mask = distance >= max(24, np.percentile(distance, 84))
    density = float(mask.mean())
    aspect = (x2 - x1) / max(1, y2 - y1)
    if density > 0.16 and aspect > 2.0:
        return "grotesk"
    return "grotesk"


def estimate_font_bold(image: Image.Image, box: tuple[int, int, int, int], background: tuple[int, int, int]) -> bool:
    arr = np.asarray(image)
    x1, y1, x2, y2 = box
    crop = arr[y1:y2, x1:x2]
    if not crop.size:
        return False
    bg = np.array(background)
    distance = np.linalg.norm(crop.astype(float) - bg, axis=2)
    mask = distance >= max(24, np.percentile(distance, 86))
    row_has_text = mask.mean(axis=1) > 0.025
    active_rows = max(1, int(row_has_text.sum()))
    glyph_density = float(mask.sum() / max(1, active_rows * mask.shape[1]))
    return glyph_density > 0.125


def estimated_contrast(color: tuple[int, int, int], background: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(color, dtype=float) - np.array(background, dtype=float)))


def estimate_text_effects(
    text_color: tuple[int, int, int],
    background: tuple[int, int, int],
    font_size: int,
) -> tuple[dict, dict]:
    bg_brightness = sum(background) / 3
    text_brightness = sum(text_color) / 3
    dark_text = text_brightness < bg_brightness
    outline_color = lighten_color(background, 0.55) if dark_text else darken_color(background, 0.55)
    shadow_color = darken_color(background, 0.18) if dark_text else lighten_color(background, 0.18)
    contrast = estimated_contrast(text_color, background)
    return (
        {
            "width": 1 if font_size >= 18 and contrast < 125 else 0,
            "fill": outline_color,
        },
        {
            "offset": 1 if font_size >= 18 else 0,
            "fill": shadow_color,
        },
    )


def lighten_color(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(channel + (255 - channel) * amount) for channel in color)


def darken_color(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(channel * (1 - amount)) for channel in color)


def clear_text_region(image: Image.Image, clear_box: tuple[int, int, int, int], style: dict) -> None:
    if clear_text_region_with_inpainting(image, clear_box, style):
        return
    clear_text_region_with_gradient(image, clear_box)


def clear_text_region_with_inpainting(image: Image.Image, clear_box: tuple[int, int, int, int], style: dict) -> bool:
    try:
        import cv2
    except Exception:
        return False

    x1, y1, x2, y2 = clear_box
    if x2 <= x1 or y2 <= y1:
        return False

    arr = np.asarray(image).copy()
    crop = arr[y1:y2, x1:x2]
    if not crop.size:
        return False

    mask = build_text_inpaint_mask(image, clear_box, style)

    coverage = float(mask.mean() / 255)
    if coverage < 0.004 or coverage > 0.34:
        return False

    kernel_size = max(3, min(5, int(min(x2 - x1, y2 - y1) * 0.035) | 1))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    full_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    inpainted = cv2.inpaint(
        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
        full_mask,
        inpaintRadius=max(3, kernel_size),
        flags=cv2.INPAINT_TELEA,
    )
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    inpainted_rgb = restore_background_texture(arr, inpainted_rgb, clear_box, mask)
    image.paste(Image.fromarray(inpainted_rgb))
    return True


def build_text_inpaint_mask(image: Image.Image, clear_box: tuple[int, int, int, int], style: dict) -> np.ndarray:
    arr = np.asarray(image).astype(np.float32)
    x1, y1, x2, y2 = clear_box
    crop = arr[y1:y2, x1:x2]
    background_patch = estimate_background_gradient(image, clear_box)
    if background_patch is None:
        background_patch = np.full(crop.shape, style.get("background") or (255, 255, 255), dtype=np.uint8)
    background_patch = background_patch.astype(np.float32)
    text = np.array(style.get("text") or (0, 0, 0), dtype=np.float32)

    bg_distance = np.linalg.norm(crop - background_patch, axis=2)
    text_distance = np.linalg.norm(crop - text, axis=2)
    crop_luma = crop.mean(axis=2)
    bg_luma = background_patch.mean(axis=2)
    text_luma = float(text.mean())

    if text_luma < float(np.median(bg_luma)):
        polarity_mask = crop_luma < bg_luma - max(10, float(np.std(bg_luma)) * 0.55)
    else:
        polarity_mask = crop_luma > bg_luma + max(10, float(np.std(bg_luma)) * 0.55)

    distance_threshold = max(22, float(np.percentile(bg_distance, 90)))
    text_threshold = max(34, min(82, float(np.percentile(text_distance, 36))))
    mask = ((bg_distance >= distance_threshold) & polarity_mask) | (text_distance <= text_threshold)

    coverage = float(mask.mean())
    if coverage > 0.34:
        strict_threshold = max(distance_threshold, float(np.percentile(bg_distance, 96)))
        mask = (bg_distance >= strict_threshold) & polarity_mask
    return mask.astype(np.uint8) * 255


def restore_background_texture(
    original: np.ndarray,
    inpainted: np.ndarray,
    clear_box: tuple[int, int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    x1, y1, x2, y2 = clear_box
    texture = estimate_surrounding_texture(original, clear_box)
    if texture is None:
        return inpainted

    result = inpainted.copy().astype(np.float32)
    h, w = mask.shape
    seed = (x1 * 73856093) ^ (y1 * 19349663) ^ (x2 * 83492791) ^ (y2 * 2654435761)
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    indices = rng.integers(0, len(texture), size=h * w)
    residual = texture[indices].reshape(h, w, 3)
    alpha = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32) / 255.0
    alpha = np.clip(alpha[..., None] * 0.72, 0, 0.72)
    result[y1:y2, x1:x2] = np.clip(result[y1:y2, x1:x2] + residual * alpha, 0, 255)
    return result.astype(np.uint8)


def estimate_surrounding_texture(original: np.ndarray, clear_box: tuple[int, int, int, int]) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None

    x1, y1, x2, y2 = clear_box
    h, w = original.shape[:2]
    margin = max(8, min(28, int(min(x2 - x1, y2 - y1) * 0.22)))
    ox1, oy1 = max(0, x1 - margin), max(0, y1 - margin)
    ox2, oy2 = min(w, x2 + margin), min(h, y2 + margin)
    outer = original[oy1:oy2, ox1:ox2].astype(np.float32)
    if not outer.size:
        return None
    ring_mask = np.ones(outer.shape[:2], dtype=bool)
    ring_mask[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
    blur = cv2.GaussianBlur(outer, (0, 0), sigmaX=2.0, sigmaY=2.0)
    residual = outer - blur
    values = residual[ring_mask]
    if len(values) < 32:
        return None
    magnitude = np.linalg.norm(values, axis=1)
    keep = magnitude <= np.percentile(magnitude, 86)
    values = values[keep]
    if len(values) < 32:
        return None
    return np.clip(values, -18, 18)


def clear_text_region_with_gradient(image: Image.Image, clear_box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = clear_box
    if x2 <= x1 or y2 <= y1:
        return
    patch_array = estimate_background_gradient(image, clear_box)
    if patch_array is None:
        patch_array = np.full((y2 - y1, x2 - x1, 3), estimate_flat_background_color(image, clear_box), dtype=np.uint8)
    patch_array = add_texture_to_gradient(np.asarray(image), patch_array, clear_box)
    patch = Image.fromarray(patch_array, "RGB")
    image.paste(patch, (x1, y1), feather_mask(patch.size))


def add_texture_to_gradient(original: np.ndarray, patch: np.ndarray, clear_box: tuple[int, int, int, int]) -> np.ndarray:
    texture = estimate_surrounding_texture(original, clear_box)
    if texture is None:
        return patch
    x1, y1, x2, y2 = clear_box
    h, w = y2 - y1, x2 - x1
    seed = (x1 * 1103515245) ^ (y1 * 12345) ^ (x2 * 2654435761) ^ y2
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    indices = rng.integers(0, len(texture), size=h * w)
    residual = texture[indices].reshape(h, w, 3)
    return np.clip(patch.astype(np.float32) + residual * 0.82, 0, 255).astype(np.uint8)


def feather_mask(size: tuple[int, int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", (width, height), 255)
    feather = max(2, min(18, int(min(width, height) * 0.12)))
    if feather <= 1:
        return mask
    edge = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(edge)
    draw.rectangle((feather, feather, width - feather, height - feather), fill=255)
    return edge.filter(ImageFilter.GaussianBlur(radius=feather / 2))


def render_fitted_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    align: str = "center",
    fill: tuple[int, int, int] = (0, 0, 0),
    max_font_size: int | None = None,
    bold: bool = True,
    outline: dict | None = None,
    shadow: dict | None = None,
    font_family: str = "humanist",
) -> None:
    x1, y1, x2, y2 = box
    max_width = max(20, x2 - x1)
    max_height = max(20, y2 - y1)
    start_size = min(max_font_size or 46, max_height)
    for font_size in range(start_size, 11, -1):
        font = load_font(font_size, bold=bold, family=font_family)
        lines = wrap_text_for_width(draw, text, font, max_width)
        line_height = int(font_size * 1.18)
        total_height = line_height * len(lines)
        if total_height <= max_height:
            y = y1 + (max_height - total_height) // 2
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                if align == "left":
                    x = x1
                elif align == "right":
                    x = x2 - line_width
                else:
                    x = x1 + (max_width - line_width) // 2
                if shadow and shadow.get("offset"):
                    offset = int(shadow.get("offset") or 1)
                    draw.text((x + offset, y + offset), line, fill=shadow.get("fill") or fill, font=font)
                draw.text(
                    (x, y),
                    line,
                    fill=fill,
                    font=font,
                    stroke_width=int((outline or {}).get("width") or 0),
                    stroke_fill=(outline or {}).get("fill") or fill,
                )
                y += line_height
            return
    font = load_font(12, bold=bold, family=font_family)
    draw.multiline_text((x1, y1), "\n".join(textwrap.wrap(text, width=18)), fill=fill, font=font, spacing=2, align=align)


def wrap_text_for_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def load_font(size: int, bold: bool = True, family: str = "humanist") -> ImageFont.ImageFont:
    candidates: list[tuple[str, int]] = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
        ("/System/Library/Fonts/Avenir Next.ttc", 2 if bold else 5),
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf", 0),
        ("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", 0),
    ]
    if family == "humanist":
        candidates = [candidates[1], candidates[0], *candidates[2:]]
    for path, index in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size, index=index)
    return ImageFont.load_default()


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def file_as_data_url(path: str) -> str:
    mime = "image/png"
    if path.lower().endswith((".jpg", ".jpeg")):
        mime = "image/jpeg"
    if path.lower().endswith(".webp"):
        mime = "image/webp"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def delete_path(path: str) -> None:
    if not path:
        return
    target = Path(path)
    try:
        if target.exists() and DATA_DIR.resolve() in target.resolve().parents:
            target.unlink()
    except OSError:
        pass


def copy_if_needed(source: str, target: str) -> None:
    if source != target:
        shutil.copyfile(source, target)
