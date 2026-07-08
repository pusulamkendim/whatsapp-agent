import asyncio
import json
import os
import hmac
import hashlib
import secrets
import re
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from fastapi import FastAPI, Request, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from sqlalchemy.exc import OperationalError, ProgrammingError
from app.config import WHATSAPP_VERIFY_TOKEN
from app.database import init_db, SessionLocal
from app.whatsapp import send_message as wa_send, extract_message as wa_extract
from app.telegram import send_message as tg_send, extract_message as tg_extract, setup_webhook as tg_setup_webhook
from app.instagram import send_message as ig_send, extract_message as ig_extract
from app import retreat_agent
from app.agent_registry import run_agent
from app.pricing import sync_llm_prices
from app.llm import (
    MODEL_OPTIONS,
    capture_llm_usage,
    parse_model_ref,
    provider_label,
    record_gemini_usage,
    run_openai_simple_chat,
    summarize_llm_usage,
)
from app.llm_usage import add_llm_usage_events, add_llm_usage_log
from app.image_localization import (
    fit_text_boxes_to_content,
    generate_output,
    prepare_asset,
    run_ocr_pipeline,
    safe_json_loads,
    store_url_images,
    store_upload,
    warm_ocr_models,
)
from app.models import (
    Agent,
    AgentKnowledgeBase,
    ChannelAccount,
    Conversation,
    DailyStat,
    Handoff,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeEmbedding,
    ImageLocalizationAsset,
    ImageLocalizationJob,
    LlmModel,
    LlmProvider,
    LlmUsageLog,
    RagQueryLog,
    Route,
)
from app.rag.indexing import index_document, index_knowledge_base, reindex_all
from app.rag.retrieval import retrieve_agent_context
from app.router import find_channel_account, resolve_route

app = FastAPI(title="WhatsApp Multi-Agent")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

ADMIN_SECRET = os.getenv("ADMIN_PASSWORD") or os.getenv("ADMIN_TOKEN") or ""
ADMIN_COOKIE = "agent_admin_session"
IS_PRODUCTION = bool(os.getenv("COOLIFY_RESOURCE_UUID")) or os.getenv("ENV", "").lower() == "production"

# Agent routing: müşteri hangi agent'a bağlı? Key: channel_account_id:external_user_id
customer_agents: dict[str, int] = {}
ocr_progress_lock = threading.Lock()
ocr_progress: dict[str, dict] = {}

# İşlenmiş mesaj ID'leri (duplicate önleme)
processed_messages: OrderedDict[str, bool] = OrderedDict()
MAX_PROCESSED = 1000


def _ocr_progress_key(job_id: int, asset_id: int) -> str:
    return f"{job_id}:{asset_id}"


def _set_ocr_progress(job_id: int, asset_id: int, step: str, message: str, meta: dict | None = None, status: str = "running") -> None:
    payload = {
        "job_id": job_id,
        "asset_id": asset_id,
        "status": status,
        "step": step,
        "message": message,
        "meta": meta or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with ocr_progress_lock:
        current = ocr_progress.get(_ocr_progress_key(job_id, asset_id), {})
        history = list(current.get("history") or [])
        if not history or history[-1].get("step") != step or history[-1].get("message") != message:
            history.append({key: payload[key] for key in ["step", "message", "meta", "updated_at"]})
        payload["history"] = history[-30:]
        ocr_progress[_ocr_progress_key(job_id, asset_id)] = payload


def _get_ocr_progress(job_id: int, asset_id: int) -> dict:
    with ocr_progress_lock:
        return dict(ocr_progress.get(_ocr_progress_key(job_id, asset_id), {
            "job_id": job_id,
            "asset_id": asset_id,
            "status": "idle",
            "step": "",
            "message": "",
            "meta": {},
            "history": [],
            "updated_at": "",
        }))

@app.on_event("startup")
def startup():
    init_db()
    threading.Thread(target=_sync_prices_on_startup, daemon=True).start()
    if os.getenv("OCR_WARMUP_ENABLE", "false").lower() in {"1", "true", "yes"}:
        threading.Thread(target=_warm_ocr_on_startup, daemon=True).start()
    # Restoran menü verisi yükle (SQLite sıfırlanırsa diye)
    from app.models import Restaurant, MenuItem
    db = SessionLocal()
    if db.query(Restaurant).count() == 0:
        import importlib, seed_menu
        seed_menu.seed(init_db_first=False)
    db.close()
    # Telegram webhook ayarla
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        base_url = os.getenv("BASE_URL", "https://agentapi.pusulamkendim.com")
        try:
            tg_setup_webhook(f"{base_url}/telegram/webhook")
        except Exception as exc:
            print(f"⚠️ Telegram webhook ayarlanamadı: {type(exc).__name__}")
    if not ADMIN_SECRET:
        message = "production'da dashboard/API kilitli" if IS_PRODUCTION else "dashboard/API auth devre dışı"
        print(f"⚠️ ADMIN_PASSWORD/ADMIN_TOKEN yok; {message}.")
    print("✅ Multi-Agent başlatıldı! (WhatsApp + Telegram)")


def _sync_prices_on_startup():
    if os.getenv("LLM_PRICE_SYNC_ON_STARTUP", "true").lower() in {"0", "false", "no"}:
        return
    try:
        result = sync_llm_prices(force=False, stale_after_hours=24)
        if result.get("updated") or result.get("errors"):
            print(f"💸 LLM price sync: {result}")
    except Exception as exc:
        print(f"⚠️ LLM price sync atlandı: {type(exc).__name__}")


def _warm_ocr_on_startup():
    try:
        result = warm_ocr_models()
        if result.get("paddleocr") or result.get("easyocr"):
            print(f"🧠 OCR warmup: {result}")
    except Exception as exc:
        print(f"⚠️ OCR warmup atlandı: {type(exc).__name__}")


@app.middleware("http")
async def protect_admin_surface(request: Request, call_next):
    path = request.url.path
    protected = path.startswith("/dashboard") or (path.startswith("/api/") and path not in {"/api/login"})
    if protected and IS_PRODUCTION and not ADMIN_SECRET:
        return JSONResponse({"error": "admin_auth_not_configured"}, status_code=503)
    if protected and not _is_admin_request(request):
        if path.startswith("/api/"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)
    return await call_next(request)


def _session_value() -> str:
    return hmac.new(ADMIN_SECRET.encode(), b"agent-admin-session", hashlib.sha256).hexdigest()


def _is_admin_request(request: Request) -> bool:
    if not ADMIN_SECRET:
        return not IS_PRODUCTION
    cookie = request.cookies.get(ADMIN_COOKIE, "")
    bearer = request.headers.get("authorization", "")
    if bearer.lower().startswith("bearer "):
        token = bearer.split(" ", 1)[1]
        return secrets.compare_digest(token, ADMIN_SECRET)
    return secrets.compare_digest(cookie, _session_value())


@app.get("/webhook")
def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """Meta webhook doğrulaması (GET)"""
    if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN:
        print("✅ Webhook doğrulandı!")
        return PlainTextResponse(content=hub_challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)


# Bugün yazanları takip (unique user + new user kontrolü)
daily_seen_users: dict[str, set] = {}  # date_str -> set of phones
all_known_users: set = set()


def save_message(
    sender: str,
    agent_type: str,
    role: str,
    message: str,
    msg_id: str = "",
    channel_type: str = "whatsapp",
    channel_account_id: int | None = None,
    agent_id: int | None = None,
):
    """Mesajı DB'ye kaydet + günlük istatistik güncelle"""
    db = SessionLocal()
    try:
        # Konuşmayı kaydet
        db.add(Conversation(
            channel_type=channel_type,
            channel_account_id=channel_account_id,
            agent_id=agent_id,
            external_user_id=sender,
            customer_phone=sender,
            agent_type=agent_type,
            role=role,
            message=message,
            external_msg_id=msg_id,
            msg_id=msg_id,
        ))

        # Günlük istatistik güncelle
        today = date.today()
        stat = db.query(DailyStat).filter(DailyStat.date == today).first()
        if not stat:
            stat = DailyStat(date=today)
            db.add(stat)

        stat.total_messages = (stat.total_messages or 0) + 1
        if role == "user":
            stat.user_messages = (stat.user_messages or 0) + 1
        else:
            stat.agent_messages = (stat.agent_messages or 0) + 1

        # Unique user takibi
        today_str = str(today)
        if today_str not in daily_seen_users:
            daily_seen_users[today_str] = set()

        if role == "user" and sender not in daily_seen_users[today_str]:
            daily_seen_users[today_str].add(sender)
            stat.unique_users = len(daily_seen_users[today_str])

            if sender not in all_known_users:
                all_known_users.add(sender)
                stat.new_users = (stat.new_users or 0) + 1

        # Agent type breakdown
        breakdown = json.loads(stat.agent_type_breakdown or "{}")
        breakdown[agent_type] = breakdown.get(agent_type, 0) + 1
        stat.agent_type_breakdown = json.dumps(breakdown)

        db.commit()
    except Exception as e:
        print(f"⚠️ DB kayıt hatası: {e}")
    finally:
        db.close()


def process_message(
    sender: str,
    clean_text: str,
    agent_id: int,
    channel_account_id: int,
    msg_id: str = "",
):
    """Mesajı arka planda işle"""
    db = SessionLocal()
    channel_account = None
    agent = None

    try:
        channel_account = db.query(ChannelAccount).filter(ChannelAccount.id == channel_account_id).first()
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not channel_account or not agent:
            print("❌ Channel account veya agent bulunamadı")
            return

        agent_type = agent.slug
        channel = channel_account.channel_type

        # Kullanıcı mesajını kaydet
        save_message(sender, agent_type, "user", clean_text, msg_id, channel, channel_account.id, agent.id)

        response = run_agent(agent, sender, clean_text, db)

        # Agent cevabını kaydet
        save_message(sender, agent_type, "agent", response, "", channel, channel_account.id, agent.id)

        print(f"🤖 [{agent_type}/{channel}] Cevap: {response}")
        _send_via_channel(channel_account, sender, response)
    except Exception as e:
        print(f"❌ Hata: {e}")
        if channel_account:
            _send_via_channel(channel_account, sender, "Bir sorun oluştu, lütfen tekrar deneyin.")
    finally:
        db.close()


def _send_via_channel(channel_account: ChannelAccount, sender: str, response: str):
    send_fns = {"whatsapp": wa_send, "telegram": tg_send, "instagram": ig_send}
    send_fn = send_fns.get(channel_account.channel_type, wa_send)
    send_fn(sender, response, channel_account=channel_account)


def _is_duplicate(msg_id: str) -> bool:
    if not msg_id:
        return False

    if msg_id in processed_messages:
        return True

    db_check = SessionLocal()
    try:
        existing = db_check.query(Conversation).filter(
            (Conversation.external_msg_id == msg_id) | (Conversation.msg_id == msg_id)
        ).first()
        return bool(existing)
    finally:
        db_check.close()


def _mark_processed(msg_id: str):
    if not msg_id:
        return
    processed_messages[msg_id] = True
    if len(processed_messages) > MAX_PROCESSED:
        processed_messages.popitem(last=False)


def _resolve_inbound(channel_type: str, account_external_id: str, sender: str, text: str, metadata: dict | None = None):
    db = SessionLocal()
    try:
        channel_account = find_channel_account(db, channel_type, account_external_id)
        if not channel_account:
            return None

        resolution = resolve_route(db, channel_account, text, metadata)
        user_key = f"{channel_account.id}:{sender}"

        if resolution and resolution.route and resolution.route.match_type != "default":
            customer_agents[user_key] = resolution.agent.id
            return {
                "channel_account_id": resolution.channel_account.id,
                "agent_id": resolution.agent.id,
                "agent_slug": resolution.agent.slug,
                "clean_text": resolution.clean_text,
            }

        if user_key in customer_agents:
            agent = db.query(Agent).filter(Agent.id == customer_agents[user_key], Agent.active == True).first()
            if agent:
                return {
                    "channel_account_id": channel_account.id,
                    "agent_id": agent.id,
                    "agent_slug": agent.slug,
                    "clean_text": text,
                }

        if not resolution:
            return None

        customer_agents[user_key] = resolution.agent.id
        return {
            "channel_account_id": resolution.channel_account.id,
            "agent_id": resolution.agent.id,
            "agent_slug": resolution.agent.slug,
            "clean_text": resolution.clean_text,
        }
    finally:
        db.close()


def _has_customer_agent(channel_type: str, account_external_id: str, sender: str) -> bool:
    db = SessionLocal()
    try:
        channel_account = find_channel_account(db, channel_type, account_external_id)
        if not channel_account:
            return False
        return f"{channel_account.id}:{sender}" in customer_agents
    finally:
        db.close()


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Gelen WhatsApp mesajlarını işle"""
    payload = await request.json()

    # Raw payload logla (debug)
    print(f"📦 Webhook payload: {json.dumps(payload, ensure_ascii=False)[:500]}")

    result = wa_extract(payload)
    if not result:
        return {"status": "ignored"}

    sender, text, msg_id, phone_number_id = result

    # Kendi business numaramızdan gelen mesajları işleme
    if sender == "905428078429":
        print(f"⏭️ Kendi numaramızdan mesaj, atlanıyor")
        return {"status": "self_message"}

    if _is_duplicate(msg_id):
        print(f"⏭️ Duplicate mesaj atlandı: {msg_id}")
        _mark_processed(msg_id)
        return {"status": "duplicate"}

    resolved = _resolve_inbound("whatsapp", phone_number_id, sender, text)
    if not resolved:
        print(f"⚠️ WhatsApp kanal/route bulunamadı: {phone_number_id}")
        return {"status": "no_route"}

    _mark_processed(msg_id)

    print(f"📩 [whatsapp/{resolved['agent_slug']}] {sender} → {resolved['clean_text']}")

    # Hemen 200 dön, mesajı arka planda işle (Meta timeout'a takılmasın)
    background_tasks.add_task(
        process_message,
        sender,
        resolved["clean_text"],
        resolved["agent_id"],
        resolved["channel_account_id"],
        msg_id,
    )

    return {"status": "ok"}


@app.post("/telegram/webhook")
async def handle_telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Gelen Telegram mesajlarını işle"""
    payload = await request.json()

    result = tg_extract(payload)
    if not result:
        return {"status": "ignored"}

    chat_id, text, msg_id = result
    msg_id = f"tg_{msg_id}"  # WhatsApp msg_id ile karışmasın

    if _is_duplicate(msg_id):
        _mark_processed(msg_id)
        return {"status": "duplicate"}

    resolved = _resolve_inbound("telegram", "", f"tg_{chat_id}", text)
    if not resolved:
        print("⚠️ Telegram kanal/route bulunamadı")
        return {"status": "no_route"}

    _mark_processed(msg_id)

    print(f"📩 [telegram/{resolved['agent_slug']}] {chat_id} → {resolved['clean_text']}")

    background_tasks.add_task(
        process_message,
        f"tg_{chat_id}",
        resolved["clean_text"],
        resolved["agent_id"],
        resolved["channel_account_id"],
        msg_id,
    )

    return {"status": "ok"}


@app.get("/instagram/webhook")
def verify_instagram_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """Instagram webhook doğrulaması (GET)"""
    if hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY_TOKEN:
        print("✅ Instagram webhook doğrulandı!")
        return PlainTextResponse(content=hub_challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)


@app.post("/instagram/webhook")
async def handle_instagram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Gelen Instagram DM mesajlarını işle (sadece reklamdan gelenler)"""
    payload = await request.json()

    result = ig_extract(payload)
    if not result:
        return {"status": "ignored"}

    sender_id, text, msg_id, is_from_ad, account_id = result
    msg_id = f"ig_{msg_id}"

    if _is_duplicate(msg_id):
        _mark_processed(msg_id)
        return {"status": "duplicate"}

    instagram_sender = f"ig_{sender_id}"
    if not is_from_ad and not _has_customer_agent("instagram", account_id, instagram_sender):
        print(f"⏭️ Instagram organik mesaj atlandı: {sender_id}")
        return {"status": "not_from_ad"}

    metadata = {"ad_source": "ADS" if is_from_ad else "", "is_from_ad": is_from_ad}
    resolved = _resolve_inbound("instagram", account_id, instagram_sender, text, metadata)
    if not resolved:
        print(f"⏭️ Instagram mesaj için kanal/route bulunamadı: {sender_id}")
        return {"status": "no_route"}

    _mark_processed(msg_id)

    print(f"📩 [instagram/{resolved['agent_slug']}] {sender_id} → {resolved['clean_text']}")

    background_tasks.add_task(
        process_message,
        instagram_sender,
        resolved["clean_text"],
        resolved["agent_id"],
        resolved["channel_account_id"],
        msg_id,
    )

    return {"status": "ok"}


@app.get("/")
def root():
    db = SessionLocal()
    try:
        agents = [a.slug for a in db.query(Agent).filter(Agent.active == True).order_by(Agent.id).all()]
        routes = db.query(Route).filter(Route.active == True).count()
    finally:
        db.close()

    return {
        "status": "running",
        "agents": agents,
        "routes": routes,
    }


@app.get("/privacy", response_class=PlainTextResponse)
def privacy_policy():
    return """Privacy Policy

Last updated: March 20, 2026

This WhatsApp agent service ("Service") is operated for the purpose of providing automated customer assistance.

Information We Collect:
- Phone number (provided via WhatsApp)
- Message content (to process your requests)

How We Use Information:
- To respond to your inquiries
- To process orders and bookings
- To improve our service

Data Retention:
- Conversation data is retained only for the duration of your session
- Order data is retained for business record purposes

Your Rights:
- You may stop interacting with the service at any time
- You may request deletion of your data by contacting us

Contact:
For questions about this privacy policy, please reach out via WhatsApp.

This service uses WhatsApp Business API by Meta Platforms, Inc.
"""


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _is_admin_request(request):
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(request, "login.html")


@app.post("/api/login")
async def login(request: Request):
    data = await request.json()
    password = data.get("password", "")
    if IS_PRODUCTION and not ADMIN_SECRET:
        return JSONResponse({"error": "admin_auth_not_configured"}, status_code=503)
    if not ADMIN_SECRET or not secrets.compare_digest(password, ADMIN_SECRET):
        return JSONResponse({"error": "invalid_credentials"}, status_code=401)

    response = JSONResponse({"ok": True})
    response.set_cookie(
        ADMIN_COOKIE,
        _session_value(),
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        max_age=60 * 60 * 12,
    )
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(ADMIN_COOKIE)
    return response


@app.get("/api/conversations")
def get_conversations(phone: str = None):
    """Konuşma geçmişini görüntüle"""
    db = SessionLocal()
    try:
        query = db.query(Conversation).order_by(Conversation.created_at.desc())
        if phone:
            query = query.filter(Conversation.customer_phone == phone)
        messages = query.limit(200).all()
        return [{
            "id": m.id,
            "phone": m.customer_phone,
            "external_user_id": m.external_user_id or m.customer_phone,
            "channel_type": m.channel_type,
            "channel_account_id": m.channel_account_id,
            "agent_id": m.agent_id,
            "agent": m.agent_type,
            "role": m.role,
            "message": m.message,
            "time": m.created_at.isoformat() if m.created_at else "",
        } for m in messages]
    finally:
        db.close()


@app.get("/api/stats")
def get_stats(days: int = 7):
    """Son X günün istatistikleri"""
    db = SessionLocal()
    try:
        stats = db.query(DailyStat).order_by(DailyStat.date.desc()).limit(days).all()
        return [{
            "date": str(s.date),
            "total_messages": s.total_messages or 0,
            "user_messages": s.user_messages or 0,
            "agent_messages": s.agent_messages or 0,
            "unique_users": s.unique_users or 0,
            "new_users": s.new_users or 0,
            "handoffs": s.handoffs or 0,
            "breakdown": json.loads(s.agent_type_breakdown or "{}"),
        } for s in stats]
    finally:
        db.close()


@app.get("/api/handoffs")
def get_handoffs():
    """Handoff kayıtlarını görüntüle"""
    db = SessionLocal()
    try:
        handoffs = db.query(Handoff).order_by(Handoff.created_at.desc()).limit(50).all()
        return [{
            "id": h.id,
            "phone": h.customer_phone,
            "name": h.customer_name,
            "summary": h.conversation_summary,
            "interest": h.interest_level,
            "status": h.status,
            "time": h.created_at.isoformat() if h.created_at else "",
        } for h in handoffs]
    finally:
        db.close()


@app.put("/api/handoffs/{handoff_id}")
async def update_handoff(handoff_id: int, request: Request):
    """Handoff durumunu güncelle"""
    data = await request.json()
    db = SessionLocal()
    try:
        h = db.query(Handoff).filter(Handoff.id == handoff_id).first()
        if not h:
            return {"error": "not found"}
        if "status" in data:
            h.status = data["status"]
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.get("/api/agents")
def get_agents():
    """Agent listesi ve ayarları"""
    db = SessionLocal()
    try:
        agents = db.query(Agent).order_by(Agent.id.asc()).all()
        return [{
            "id": a.id,
            "slug": a.slug,
            "type": a.type,
            "name": a.name,
            "active": a.active,
            "model": a.model,
            "fallback_model": a.fallback_model or "",
            "temperature": a.temperature if a.temperature is not None else 0.7,
            "max_tokens": a.max_tokens,
            "timeout_seconds": a.timeout_seconds or 60,
            "daily_budget_limit": a.daily_budget_limit,
            "monthly_budget_limit": a.monthly_budget_limit,
            "failover_enabled": a.failover_enabled,
            "model_label": provider_label(a.model),
            "system_prompt": a.system_prompt or _default_prompt_preview(a.slug),
            "knowledge_base": a.knowledge_base or "",
            "rag_top_k": a.rag_top_k or 20,
            "rag_final_chunks": a.rag_final_chunks or 6,
            "rag_min_score": a.rag_min_score,
            "rag_max_context_chars": a.rag_max_context_chars or 12000,
            "rag_hybrid_search": a.rag_hybrid_search,
            "rag_rerank_enabled": a.rag_rerank_enabled,
            "rag_query_rewrite_enabled": a.rag_query_rewrite_enabled,
            "rag_embedding_model": a.rag_embedding_model or "",
        } for a in agents]
    finally:
        db.close()


@app.post("/api/agents")
async def create_agent(request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        slug = (data.get("slug") or "").strip().lower()
        if not slug:
            return JSONResponse({"error": "slug_required"}, status_code=400)
        if db.query(Agent).filter(Agent.slug == slug).first():
            return JSONResponse({"error": "slug_exists"}, status_code=400)
        agent = Agent(
            slug=slug,
            name=data.get("name") or slug,
            type=data.get("type") or "generic_prompt",
            model=data.get("model") or "gemini:gemini-2.5-flash",
            fallback_model=data.get("fallback_model", ""),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
            timeout_seconds=data.get("timeout_seconds", 60),
            daily_budget_limit=data.get("daily_budget_limit"),
            monthly_budget_limit=data.get("monthly_budget_limit"),
            failover_enabled=data.get("failover_enabled", True),
            system_prompt=data.get("system_prompt", ""),
            knowledge_base=data.get("knowledge_base", ""),
            rag_top_k=data.get("rag_top_k", 20),
            rag_final_chunks=data.get("rag_final_chunks", 6),
            rag_min_score=data.get("rag_min_score"),
            rag_max_context_chars=data.get("rag_max_context_chars", 12000),
            rag_hybrid_search=data.get("rag_hybrid_search", True),
            rag_rerank_enabled=data.get("rag_rerank_enabled", False),
            rag_query_rewrite_enabled=data.get("rag_query_rewrite_enabled", False),
            rag_embedding_model=data.get("rag_embedding_model", ""),
            active=data.get("active", True),
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return {"ok": True, "id": agent.id}
    finally:
        db.close()


@app.put("/api/agents/{agent_type}/config")
async def update_agent_config(agent_type: str, request: Request):
    """Agent ayarlarını güncelle"""
    data = await request.json()
    db = SessionLocal()
    try:
        agent = db.query(Agent).filter(Agent.slug == agent_type).first()
        if not agent:
            return {"error": "unknown agent"}
        if "active" in data:
            agent.active = data["active"]
        if "name" in data:
            agent.name = data["name"]
        if "model" in data:
            agent.model = data["model"]
        if "fallback_model" in data:
            agent.fallback_model = data["fallback_model"] or ""
        if "temperature" in data:
            agent.temperature = data["temperature"]
        if "max_tokens" in data:
            agent.max_tokens = data["max_tokens"] or None
        if "timeout_seconds" in data:
            agent.timeout_seconds = data["timeout_seconds"] or 60
        if "daily_budget_limit" in data:
            agent.daily_budget_limit = data["daily_budget_limit"] or None
        if "monthly_budget_limit" in data:
            agent.monthly_budget_limit = data["monthly_budget_limit"] or None
        if "failover_enabled" in data:
            agent.failover_enabled = data["failover_enabled"]
        if "system_prompt" in data:
            agent.system_prompt = data["system_prompt"]
            if agent.slug == "retreat":
                retreat_agent.SYSTEM_PROMPT = data["system_prompt"]
        if "knowledge_base" in data:
            agent.knowledge_base = data["knowledge_base"]
        for field in [
            "rag_top_k",
            "rag_final_chunks",
            "rag_min_score",
            "rag_max_context_chars",
            "rag_hybrid_search",
            "rag_rerank_enabled",
            "rag_query_rewrite_enabled",
            "rag_embedding_model",
        ]:
            if field in data:
                setattr(agent, field, data[field] or "" if field == "rag_embedding_model" else data[field])
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: int):
    db = SessionLocal()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            return JSONResponse({"error": "not_found"}, status_code=404)
        agent.active = False
        db.query(Route).filter(Route.agent_id == agent.id).update({"active": False})
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/api/agents/{agent_id}/test")
async def test_agent(agent_id: int, request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            return JSONResponse({"error": "not_found"}, status_code=404)
        if data.get("dry_run", True):
            return {
                "ok": True,
                "agent": agent.slug,
                "message": data.get("message", ""),
                "response": f"[dry-run] {agent.name} mesajı alırdı.",
            }
        response = run_agent(agent, f"test_{agent.id}", data.get("message", "Merhaba"), db)
        return {"ok": True, "agent": agent.slug, "response": response}
    finally:
        db.close()


def _default_prompt_preview(agent_slug: str) -> str:
    if agent_slug == "retreat":
        return retreat_agent.SYSTEM_PROMPT[:2000]
    if agent_slug == "restaurant":
        from app.agent import get_system_prompt
        return get_system_prompt("Lezzet Durağı")
    return ""


@app.get("/api/llm/models")
def get_llm_models():
    db = SessionLocal()
    try:
        models = db.query(LlmModel).join(LlmProvider).order_by(
            LlmProvider.slug.asc(),
            LlmModel.display_name.asc(),
        ).all()
        if not models:
            return MODEL_OPTIONS
        return [_llm_model_payload(model) for model in models if model.active and model.provider and model.provider.active]
    finally:
        db.close()


@app.get("/api/llm/providers")
def list_llm_providers():
    db = SessionLocal()
    try:
        return [_llm_provider_payload(provider) for provider in db.query(LlmProvider).order_by(LlmProvider.slug.asc()).all()]
    finally:
        db.close()


@app.post("/api/llm/providers")
async def create_llm_provider(request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        slug = _slugify(data.get("slug") or data.get("name"))
        if db.query(LlmProvider).filter(LlmProvider.slug == slug).first():
            return JSONResponse({"error": "provider_exists"}, status_code=400)
        provider = LlmProvider(
            slug=slug,
            name=data.get("name") or slug,
            base_url=data.get("base_url", ""),
            api_key_env=data.get("api_key_env", ""),
            active=data.get("active", True),
        )
        db.add(provider)
        db.commit()
        db.refresh(provider)
        return {"ok": True, "provider": _llm_provider_payload(provider)}
    finally:
        db.close()


@app.put("/api/llm/providers/{provider_id}")
async def update_llm_provider(provider_id: int, request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        provider = db.query(LlmProvider).filter(LlmProvider.id == provider_id).first()
        if not provider:
            return JSONResponse({"error": "not_found"}, status_code=404)
        for field in ["name", "base_url", "api_key_env", "active"]:
            if field in data:
                setattr(provider, field, data[field])
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/api/llm/models")
async def create_llm_model(request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        provider = db.query(LlmProvider).filter(LlmProvider.id == data.get("provider_id")).first()
        if not provider:
            return JSONResponse({"error": "provider_not_found"}, status_code=404)
        model_ref = data.get("model_ref") or f"{provider.slug}:{data.get('slug', '').strip()}"
        if db.query(LlmModel).filter(LlmModel.model_ref == model_ref).first():
            return JSONResponse({"error": "model_exists"}, status_code=400)
        model = LlmModel(provider_id=provider.id)
        _apply_llm_model_payload(model, data, provider)
        db.add(model)
        db.commit()
        db.refresh(model)
        return {"ok": True, "model": _llm_model_payload(model)}
    finally:
        db.close()


@app.put("/api/llm/models/{model_id}")
async def update_llm_model(model_id: int, request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        model = db.query(LlmModel).filter(LlmModel.id == model_id).first()
        if not model:
            return JSONResponse({"error": "not_found"}, status_code=404)
        provider = db.query(LlmProvider).filter(LlmProvider.id == data.get("provider_id", model.provider_id)).first()
        if not provider:
            return JSONResponse({"error": "provider_not_found"}, status_code=404)
        _apply_llm_model_payload(model, data, provider)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.delete("/api/llm/models/{model_id}")
def delete_llm_model(model_id: int):
    db = SessionLocal()
    try:
        model = db.query(LlmModel).filter(LlmModel.id == model_id).first()
        if not model:
            return JSONResponse({"error": "not_found"}, status_code=404)
        model.active = False
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/api/llm/prices/sync")
def sync_llm_model_prices():
    try:
        return {"ok": True, "result": sync_llm_prices(force=True)}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)[:800]}, status_code=500)


@app.post("/api/llm/models/test")
async def test_llm_model(request: Request):
    data = await request.json()
    model_ref = data.get("model_ref") or "gemini:gemini-2.5-flash"
    message = data.get("message") or "Reply with OK only."
    started = time.perf_counter()
    usage_events = []
    try:
        with capture_llm_usage() as usage_events:
            if parse_model_ref(model_ref)[0] == "gemini":
                from google.genai import types
                from app.llm import GEMINI_CLIENT

                response = GEMINI_CLIENT.models.generate_content(
                    model=parse_model_ref(model_ref)[1],
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=message)])],
                )
                record_gemini_usage(model_ref, response)
                text = response.candidates[0].content.parts[0].text
            else:
                text = run_openai_simple_chat(model_ref, [{"role": "user", "content": message}], 0)
            usage = summarize_llm_usage(usage_events)
        db = SessionLocal()
        try:
            if usage_events:
                add_llm_usage_events(
                    db,
                    usage_events,
                    source="llm-page",
                    operation="model-test",
                    latency_ms=_elapsed_ms(started),
                )
            else:
                add_llm_usage_log(
                    db,
                    model_ref=model_ref,
                    success=True,
                    source="llm-page",
                    operation="model-test",
                    latency_ms=_elapsed_ms(started),
                )
            db.commit()
        finally:
            db.close()
        return {"ok": True, "model_ref": model_ref, "response": text, "usage": usage}
    except Exception as exc:
        db = SessionLocal()
        try:
            if usage_events:
                add_llm_usage_events(
                    db,
                    usage_events,
                    source="llm-page",
                    operation="model-test",
                    latency_ms=_elapsed_ms(started),
                    success=False,
                    error=exc,
                )
            else:
                add_llm_usage_log(
                    db,
                    model_ref=model_ref,
                    success=False,
                    source="llm-page",
                    operation="model-test",
                    latency_ms=_elapsed_ms(started),
                    error=exc,
                )
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
        return JSONResponse({"ok": False, "model_ref": model_ref, "error": str(exc)[:800]}, status_code=400)


@app.get("/api/llm/usage")
def get_llm_usage(
    limit: int = Query(20, ge=1, le=500),
    days: int = Query(7, ge=1, le=90),
):
    db = SessionLocal()
    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        base_query = db.query(LlmUsageLog).filter(LlmUsageLog.created_at >= since)
        rows = base_query.order_by(LlmUsageLog.created_at.desc()).limit(limit).all()
        all_rows = base_query.all()
        total = len(all_rows)
        errors = sum(1 for row in all_rows if not row.success)
        total_tokens = sum(row.total_tokens or 0 for row in all_rows)
        prompt_tokens = sum(row.prompt_tokens or 0 for row in all_rows)
        completion_tokens = sum(row.completion_tokens or 0 for row in all_rows)
        estimated_cost = sum(row.estimated_cost or 0 for row in all_rows)
        latencies = [row.latency_ms for row in all_rows if row.latency_ms is not None]
        return {
            "summary": {
                "total_calls": total,
                "errors": errors,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost": estimated_cost,
                "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
                "success_rate": round(((total - errors) / total) * 100, 1) if total else 0,
                "days": days,
            },
            "by_source": _usage_breakdown(all_rows, "source"),
            "by_provider": _usage_breakdown(all_rows, "provider"),
            "by_model": _usage_breakdown(all_rows, "model_ref"),
            "logs": [{
                "id": row.id,
                "agent_id": row.agent_id,
                "agent_slug": row.agent.slug if row.agent else "",
                "source": row.source or "",
                "operation": row.operation or "",
                "provider": row.provider,
                "model_ref": row.model_ref,
                "generation_id": row.generation_id or "",
                "actual_model_ref": row.actual_model_ref or "",
                "actual_provider": row.actual_provider or "",
                "router": row.router or "",
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "total_tokens": row.total_tokens,
                "estimated_cost": row.estimated_cost,
                "actual_cost": row.actual_cost,
                "latency_ms": row.latency_ms,
                "success": row.success,
                "error_code": row.error_code,
                "error_message": row.error_message,
                "created_at": row.created_at.isoformat() if row.created_at else "",
            } for row in rows],
        }
    finally:
        db.close()


def _usage_breakdown(rows: list[LlmUsageLog], field: str) -> list[dict]:
    grouped: dict[str, list[LlmUsageLog]] = {}
    for row in rows:
        key = getattr(row, field) or "unknown"
        grouped.setdefault(key, []).append(row)

    items = []
    for key, group in grouped.items():
        total = len(group)
        errors = sum(1 for row in group if not row.success)
        total_tokens = sum(row.total_tokens or 0 for row in group)
        prompt_tokens = sum(row.prompt_tokens or 0 for row in group)
        completion_tokens = sum(row.completion_tokens or 0 for row in group)
        estimated_cost = sum(row.estimated_cost or 0 for row in group)
        latencies = [row.latency_ms for row in group if row.latency_ms is not None]
        items.append({
            "key": key,
            "calls": total,
            "errors": errors,
            "success_rate": round(((total - errors) / total) * 100, 1) if total else 0,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost": estimated_cost,
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        })
    return sorted(items, key=lambda item: (item["total_tokens"], item["calls"]), reverse=True)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _llm_provider_payload(provider: LlmProvider) -> dict:
    return {
        "id": provider.id,
        "slug": provider.slug,
        "name": provider.name,
        "base_url": provider.base_url or "",
        "api_key_env": provider.api_key_env or "",
        "api_key_status": "set" if provider.api_key_env and os.getenv(provider.api_key_env) else "missing",
        "active": provider.active,
    }


def _llm_model_payload(model: LlmModel) -> dict:
    provider = model.provider
    return {
        "id": model.id,
        "provider_id": model.provider_id,
        "provider": provider.slug if provider else parse_model_ref(model.model_ref)[0],
        "label": model.display_name,
        "display_name": model.display_name,
        "slug": model.slug,
        "model": model.model_ref,
        "model_ref": model.model_ref,
        "env": provider.api_key_env if provider else "",
        "supports_tools": model.supports_tools,
        "supports_vision": model.supports_vision,
        "context_window": model.context_window,
        "input_price": model.input_price,
        "output_price": model.output_price,
        "pricing_source": model.pricing_source or "",
        "pricing_checked_at": model.pricing_checked_at.isoformat() if model.pricing_checked_at else "",
        "pricing_sync_error": model.pricing_sync_error or "",
        "rate_limit_rpm": model.rate_limit_rpm,
        "rate_limit_tpm": model.rate_limit_tpm,
        "active": model.active,
        "is_default": model.is_default,
        "notes": model.notes or "",
    }


def _apply_llm_model_payload(model: LlmModel, data: dict, provider: LlmProvider):
    slug = (data.get("slug") or model.slug or "").strip()
    model_ref = data.get("model_ref") or f"{provider.slug}:{slug}"
    model.provider_id = provider.id
    model.slug = slug or model_ref.split(":", 1)[-1]
    model.display_name = data.get("display_name") or data.get("label") or model.slug
    model.model_ref = model_ref
    model.supports_tools = data.get("supports_tools", model.supports_tools or False)
    model.supports_vision = data.get("supports_vision", model.supports_vision or False)
    model.context_window = data.get("context_window") or None
    model.input_price = data.get("input_price") if data.get("input_price") is not None else None
    model.output_price = data.get("output_price") if data.get("output_price") is not None else None
    model.rate_limit_rpm = data.get("rate_limit_rpm") or None
    model.rate_limit_tpm = data.get("rate_limit_tpm") or None
    model.active = data.get("active", model.active if model.active is not None else True)
    model.is_default = data.get("is_default", model.is_default or False)
    model.notes = data.get("notes", model.notes or "")


def _mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••"
    return f"{value[:2]}••••{value[-2:]}"


def _mask_credentials(credentials_json: str | None) -> str:
    if not credentials_json:
        return "{}"
    try:
        data = json.loads(credentials_json)
    except json.JSONDecodeError:
        return "{}"
    masked = {}
    for key, value in data.items():
        if key.endswith("_env") or key in {"phone_number_id", "bot_username"}:
            masked[key] = value
        else:
            masked[key] = _mask_secret(str(value))
    return json.dumps(masked, ensure_ascii=False)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or secrets.token_hex(4)


def _knowledge_base_payload(kb: KnowledgeBase) -> dict:
    active_docs = [doc for doc in kb.documents if doc.active]
    active_links = [link for link in kb.agent_links if link.active and link.agent]
    return {
        "id": kb.id,
        "slug": kb.slug,
        "name": kb.name,
        "description": kb.description or "",
        "active": kb.active,
        "document_count": len(active_docs),
        "character_count": sum(len(doc.content or "") for doc in active_docs),
        "agents": [{
            "id": link.agent.id,
            "slug": link.agent.slug,
            "name": link.agent.name,
            "priority": link.priority,
        } for link in sorted(active_links, key=lambda item: (item.priority or 100, item.id))],
        "created_at": kb.created_at.isoformat() if kb.created_at else "",
        "updated_at": kb.updated_at.isoformat() if kb.updated_at else "",
    }


def _document_payload(doc: KnowledgeDocument) -> dict:
    active_chunks = [chunk for chunk in doc.chunks if chunk.active]
    active_embeddings = [
        embedding
        for chunk in active_chunks
        for embedding in chunk.embeddings
        if embedding.status == "ready"
    ]
    failed_embeddings = [
        embedding
        for chunk in active_chunks
        for embedding in chunk.embeddings
        if embedding.status == "failed"
    ]
    indexed_at_values = [
        embedding.updated_at or embedding.created_at
        for chunk in active_chunks
        for embedding in chunk.embeddings
        if embedding.status == "ready" and (embedding.updated_at or embedding.created_at)
    ]
    last_indexed_at = max(indexed_at_values).isoformat() if indexed_at_values else ""
    if active_chunks and len(active_embeddings) >= len(active_chunks):
        rag_status = "indexed"
    elif failed_embeddings:
        rag_status = "failed"
    elif active_chunks:
        rag_status = "pending"
    else:
        rag_status = "not_indexed"
    return {
        "id": doc.id,
        "knowledge_base_id": doc.knowledge_base_id,
        "filename": doc.filename,
        "content_type": doc.content_type,
        "source_type": doc.source_type,
        "active": doc.active,
        "character_count": len(doc.content or ""),
        "chunk_count": len(active_chunks),
        "rag_status": rag_status,
        "rag_error": failed_embeddings[-1].error_message if failed_embeddings else "",
        "last_indexed_at": last_indexed_at,
        "preview": (doc.content or "")[:280],
        "created_at": doc.created_at.isoformat() if doc.created_at else "",
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else "",
    }


@app.get("/api/channels")
def get_channels():
    db = SessionLocal()
    try:
        accounts = db.query(ChannelAccount).order_by(ChannelAccount.id.asc()).all()
        return [{
            "id": c.id,
            "channel_type": c.channel_type,
            "name": c.name,
            "external_id": c.external_id,
            "display_identifier": c.display_identifier,
            "credentials_json": _mask_credentials(c.credentials_json),
            "credentials_raw_editable": False,
            "webhook_secret": _mask_secret(c.webhook_secret),
            "active": c.active,
            "created_at": c.created_at.isoformat() if c.created_at else "",
            "updated_at": c.updated_at.isoformat() if c.updated_at else "",
        } for c in accounts]
    finally:
        db.close()


@app.post("/api/channels")
async def create_channel(request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        account = ChannelAccount(
            channel_type=data["channel_type"],
            name=data["name"],
            external_id=data["external_id"],
            display_identifier=data.get("display_identifier", ""),
            credentials_json=data.get("credentials_json", "{}"),
            webhook_secret=data.get("webhook_secret", ""),
            active=data.get("active", True),
        )
        db.add(account)
        db.commit()
        db.refresh(account)
        return {"ok": True, "id": account.id}
    finally:
        db.close()


@app.put("/api/channels/{channel_id}")
async def update_channel(channel_id: int, request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        account = db.query(ChannelAccount).filter(ChannelAccount.id == channel_id).first()
        if not account:
            return {"error": "not found"}
        for field in ["channel_type", "name", "external_id", "display_identifier", "credentials_json", "webhook_secret", "active"]:
            if field in data and data[field] != "__KEEP__":
                setattr(account, field, data[field])
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.delete("/api/channels/{channel_id}")
def delete_channel(channel_id: int):
    db = SessionLocal()
    try:
        account = db.query(ChannelAccount).filter(ChannelAccount.id == channel_id).first()
        if not account:
            return JSONResponse({"error": "not_found"}, status_code=404)
        account.active = False
        db.query(Route).filter(Route.channel_account_id == account.id).update({"active": False})
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.get("/api/routes")
def get_routes():
    db = SessionLocal()
    try:
        routes = db.query(Route).order_by(Route.channel_account_id.asc(), Route.priority.asc()).all()
        return [{
            "id": r.id,
            "channel_account_id": r.channel_account_id,
            "channel_name": r.channel_account.name if r.channel_account else "",
            "agent_id": r.agent_id,
            "agent_slug": r.agent.slug if r.agent else "",
            "priority": r.priority,
            "match_type": r.match_type,
            "match_value": r.match_value,
            "active": r.active,
        } for r in routes]
    finally:
        db.close()


@app.post("/api/routes")
async def create_route(request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        route = Route(
            channel_account_id=data["channel_account_id"],
            agent_id=data["agent_id"],
            priority=data.get("priority", 100),
            match_type=data.get("match_type", "default"),
            match_value=data.get("match_value", ""),
            active=data.get("active", True),
        )
        db.add(route)
        db.commit()
        db.refresh(route)
        _clear_route_cache(route.channel_account_id)
        return {"ok": True, "id": route.id}
    finally:
        db.close()


@app.put("/api/routes/{route_id}")
async def update_route(route_id: int, request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        route = db.query(Route).filter(Route.id == route_id).first()
        if not route:
            return {"error": "not found"}
        for field in ["channel_account_id", "agent_id", "priority", "match_type", "match_value", "active"]:
            if field in data:
                setattr(route, field, data[field])
        db.commit()
        _clear_route_cache(route.channel_account_id)
        return {"ok": True}
    finally:
        db.close()


@app.delete("/api/routes/{route_id}")
def delete_route(route_id: int):
    db = SessionLocal()
    try:
        route = db.query(Route).filter(Route.id == route_id).first()
        if not route:
            return JSONResponse({"error": "not_found"}, status_code=404)
        route.active = False
        db.commit()
        _clear_route_cache(route.channel_account_id)
        return {"ok": True}
    finally:
        db.close()


def _clear_route_cache(channel_account_id: int | None = None):
    if channel_account_id is None:
        customer_agents.clear()
        return
    prefix = f"{channel_account_id}:"
    for key in list(customer_agents.keys()):
        if key.startswith(prefix):
            customer_agents.pop(key, None)


@app.post("/api/routes/simulate")
async def simulate_route(request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        account = db.query(ChannelAccount).filter(ChannelAccount.id == data.get("channel_account_id")).first()
        if not account:
            return JSONResponse({"error": "channel_not_found"}, status_code=404)
        resolution = resolve_route(db, account, data.get("message", ""), data.get("metadata") or {})
        if not resolution:
            return {"ok": True, "matched": False}
        return {
            "ok": True,
            "matched": True,
            "agent_id": resolution.agent.id,
            "agent_slug": resolution.agent.slug,
            "route_id": resolution.route.id if resolution.route else None,
            "match_type": resolution.route.match_type if resolution.route else "",
            "match_value": resolution.route.match_value if resolution.route else "",
            "clean_text": resolution.clean_text,
        }
    finally:
        db.close()


@app.get("/api/knowledge-bases")
def list_knowledge_bases():
    try:
        return _list_knowledge_bases()
    except (OperationalError, ProgrammingError):
        init_db()
        return _list_knowledge_bases()


def _list_knowledge_bases():
    db = SessionLocal()
    try:
        bases = db.query(KnowledgeBase).order_by(KnowledgeBase.active.desc(), KnowledgeBase.id.asc()).all()
        return [_knowledge_base_payload(kb) for kb in bases]
    finally:
        db.close()


@app.post("/api/knowledge-bases")
async def create_knowledge_base(request: Request):
    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name_required"}, status_code=400)

    db = SessionLocal()
    try:
        base_slug = _slugify(data.get("slug") or name)
        slug = base_slug
        suffix = 2
        while db.query(KnowledgeBase).filter(KnowledgeBase.slug == slug).first():
            slug = f"{base_slug}-{suffix}"
            suffix += 1

        kb = KnowledgeBase(
            slug=slug,
            name=name,
            description=data.get("description", ""),
            active=data.get("active", True),
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)
        return {"ok": True, "knowledge_base": _knowledge_base_payload(kb)}
    finally:
        db.close()


@app.put("/api/knowledge-bases/{kb_id}")
async def update_knowledge_base(kb_id: int, request: Request):
    data = await request.json()
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            return JSONResponse({"error": "not_found"}, status_code=404)
        for field in ["name", "description", "active"]:
            if field in data:
                setattr(kb, field, data[field])
        db.commit()
        db.refresh(kb)
        return {"ok": True, "knowledge_base": _knowledge_base_payload(kb)}
    finally:
        db.close()


@app.delete("/api/knowledge-bases/{kb_id}")
def delete_knowledge_base(kb_id: int):
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            return JSONResponse({"error": "not_found"}, status_code=404)
        kb.active = False
        db.query(AgentKnowledgeBase).filter(AgentKnowledgeBase.knowledge_base_id == kb.id).update({"active": False})
        db.query(KnowledgeChunk).filter(KnowledgeChunk.knowledge_base_id == kb.id).update({"active": False})
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.get("/api/knowledge-bases/{kb_id}/documents")
def list_knowledge_documents(kb_id: int):
    db = SessionLocal()
    try:
        docs = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.knowledge_base_id == kb_id,
            KnowledgeDocument.active == True,
        ).order_by(KnowledgeDocument.id.asc()).all()
        return [_document_payload(doc) for doc in docs]
    finally:
        db.close()


@app.post("/api/knowledge-bases/{kb_id}/documents")
async def upload_knowledge_documents(kb_id: int, files: list[UploadFile] = File(...)):
    max_bytes = 2 * 1024 * 1024
    allowed_extensions = {".txt", ".md", ".markdown", ".csv", ".json"}
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.active == True).first()
        if not kb:
            return JSONResponse({"error": "not_found"}, status_code=404)

        created = []
        for upload in files:
            filename = os.path.basename(upload.filename or "document.txt")
            extension = os.path.splitext(filename)[1].lower()
            if extension and extension not in allowed_extensions:
                return JSONResponse({"error": "unsupported_file_type", "filename": filename}, status_code=400)

            raw = await upload.read()
            if len(raw) > max_bytes:
                return JSONResponse({"error": "file_too_large", "filename": filename}, status_code=400)

            content = raw.decode("utf-8", errors="replace").strip()
            if not content:
                continue

            doc = KnowledgeDocument(
                knowledge_base_id=kb.id,
                filename=filename,
                content_type=upload.content_type or "text/plain",
                content=content,
                source_type="upload",
                active=True,
            )
            db.add(doc)
            db.flush()
            index_document(doc, db)
            created.append(_document_payload(doc))

        db.commit()
        return {"ok": True, "documents": created}
    finally:
        db.close()


@app.post("/api/knowledge-bases/{kb_id}/documents/text")
async def create_knowledge_text_document(kb_id: int, request: Request):
    data = await request.json()
    content = (data.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "content_required"}, status_code=400)

    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.active == True).first()
        if not kb:
            return JSONResponse({"error": "not_found"}, status_code=404)
        doc = KnowledgeDocument(
            knowledge_base_id=kb.id,
            filename=data.get("filename") or "manual-note.md",
            content_type="text/markdown",
            content=content,
            source_type="manual",
            active=True,
        )
        db.add(doc)
        db.flush()
        index_document(doc, db)
        db.commit()
        db.refresh(doc)
        return {"ok": True, "document": _document_payload(doc)}
    finally:
        db.close()


@app.delete("/api/knowledge-documents/{document_id}")
def delete_knowledge_document(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
        if not doc:
            return JSONResponse({"error": "not_found"}, status_code=404)
        doc.active = False
        db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc.id).update({"active": False})
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/api/knowledge-bases/{kb_id}/reindex")
def reindex_knowledge_base(kb_id: int):
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.active == True).first()
        if not kb:
            return JSONResponse({"error": "not_found"}, status_code=404)
        result = index_knowledge_base(kb.id, db)
        db.commit()
        return result
    except Exception as exc:
        db.rollback()
        return JSONResponse({"error": "reindex_failed", "detail": str(exc)[:1000]}, status_code=500)
    finally:
        db.close()


@app.post("/api/rag/reindex-all")
async def reindex_all_knowledge(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    db = SessionLocal()
    try:
        result = reindex_all(db, model_ref=data.get("embedding_model"))
        db.commit()
        return result
    except Exception as exc:
        db.rollback()
        return JSONResponse({"error": "reindex_all_failed", "detail": str(exc)[:1000]}, status_code=500)
    finally:
        db.close()


@app.post("/api/agents/{agent_id}/rag/search")
async def search_agent_rag(agent_id: int, request: Request):
    data = await request.json()
    query = (data.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query_required"}, status_code=400)

    db = SessionLocal()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.active == True).first()
        if not agent:
            return JSONResponse({"error": "agent_not_found"}, status_code=404)
        result = retrieve_agent_context(
            agent,
            query,
            db,
            external_user_id=data.get("external_user_id", "rag-test"),
            final_chunks=int(data.get("final_chunks") or agent.rag_final_chunks or 6),
            top_k=int(data.get("top_k") or agent.rag_top_k or 20),
            max_context_chars=int(data.get("max_context_chars") or agent.rag_max_context_chars or 12000),
            min_score=data.get("min_score", agent.rag_min_score),
            hybrid_search=data.get("hybrid_search", agent.rag_hybrid_search),
            rerank_enabled=data.get("rerank_enabled", agent.rag_rerank_enabled),
            query_rewrite_enabled=data.get("query_rewrite_enabled", agent.rag_query_rewrite_enabled),
            model_ref=data.get("embedding_model") or agent.rag_embedding_model or None,
        )
        db.commit()
        return {
            "ok": True,
            "query": result.query,
            "rewritten_query": result.rewritten_query,
            "context": result.context,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "document_id": chunk.document_id,
                    "title_path": chunk.title_path,
                    "score": chunk.score,
                    "vector_score": chunk.vector_score,
                    "keyword_score": chunk.keyword_score,
                    "preview": chunk.content[:500],
                }
                for chunk in result.chunks
            ],
        }
    except Exception as exc:
        db.rollback()
        return JSONResponse({"error": "rag_search_failed", "detail": str(exc)[:1000]}, status_code=500)
    finally:
        db.close()


@app.put("/api/knowledge-bases/{kb_id}/agents")
async def update_knowledge_agents(kb_id: int, request: Request):
    data = await request.json()
    agent_ids = {int(agent_id) for agent_id in data.get("agent_ids", [])}
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            return JSONResponse({"error": "not_found"}, status_code=404)

        existing = db.query(AgentKnowledgeBase).filter(AgentKnowledgeBase.knowledge_base_id == kb.id).all()
        links_by_agent = {link.agent_id: link for link in existing}
        for link in existing:
            link.active = False

        for index, agent_id in enumerate(sorted(agent_ids)):
            if not db.query(Agent).filter(Agent.id == agent_id).first():
                continue
            link = links_by_agent.get(agent_id)
            if link:
                link.active = True
                link.priority = 100 + index
            else:
                db.add(AgentKnowledgeBase(
                    agent_id=agent_id,
                    knowledge_base_id=kb.id,
                    priority=100 + index,
                    active=True,
                ))

        db.commit()
        db.refresh(kb)
        return {"ok": True, "knowledge_base": _knowledge_base_payload(kb)}
    finally:
        db.close()


@app.get("/api/knowledge")
def get_knowledge():
    """Legacy endpoint: default retreat knowledge içeriği."""
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.slug == "retreat-default").first()
        if not kb:
            return {"content": ""}
        docs = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.knowledge_base_id == kb.id,
            KnowledgeDocument.active == True,
        ).order_by(KnowledgeDocument.id.asc()).all()
        return {"content": "\n\n---\n\n".join(doc.content for doc in docs)}
    finally:
        db.close()


@app.put("/api/knowledge")
async def update_knowledge(request: Request):
    """Legacy endpoint: default retreat knowledge manuel dokümanına yazar."""
    data = await request.json()
    db = SessionLocal()
    try:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.slug == "retreat-default").first()
        if not kb:
            kb = KnowledgeBase(slug="retreat-default", name="Retreat Knowledge", active=True)
            db.add(kb)
            db.flush()
        doc = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.knowledge_base_id == kb.id,
            KnowledgeDocument.filename == "legacy-editor.md",
        ).first()
        if not doc:
            doc = KnowledgeDocument(
                knowledge_base_id=kb.id,
                filename="legacy-editor.md",
                content_type="text/markdown",
                content=data["content"],
                source_type="manual",
                active=True,
            )
            db.add(doc)
        else:
            doc.content = data["content"]
            doc.active = True
        db.commit()
        retreat_agent.KNOWLEDGE_BASE = data["content"]
        retreat_agent.SYSTEM_PROMPT = retreat_agent.build_system_prompt(retreat_agent.KNOWLEDGE_BASE)
        return {"ok": True}
    finally:
        db.close()


# ============ IMAGE LOCALIZATION API ============

@app.post("/api/image-localization/jobs")
async def create_image_localization_job(request: Request, files: list[UploadFile] = File(default=[])):
    form = await request.form()
    urls = [value.strip() for value in form.getlist("urls") if isinstance(value, str) and value.strip()]
    if not files and not urls:
        return JSONResponse({"error": "files_or_urls_required"}, status_code=400)

    db = SessionLocal()
    try:
        job = ImageLocalizationJob(target_language="tr", status="uploaded")
        db.add(job)
        db.flush()

        assets = []
        for upload in files:
            raw = await upload.read()
            asset = _create_image_localization_asset_from_source(
                job.id,
                upload.filename or "image.png",
                upload.content_type or "image/png",
                lambda: store_upload(upload.filename or "image.png", upload.content_type or "image/png", raw),
            )
            db.add(asset)
            db.flush()
            assets.append(asset)

        for url in urls:
            try:
                stored_images = store_url_images(url)
            except ValueError as exc:
                asset = ImageLocalizationAsset(
                    job_id=job.id,
                    filename=os.path.basename(url) or url[:120] or "remote-image",
                    content_type="image/png",
                    original_path="",
                    status="failed",
                    error_message=str(exc),
                )
                db.add(asset)
                db.flush()
                assets.append(asset)
                continue
            except Exception as exc:
                asset = ImageLocalizationAsset(
                    job_id=job.id,
                    filename=os.path.basename(url) or url[:120] or "remote-image",
                    content_type="image/png",
                    original_path="",
                    status="failed",
                    error_message=f"{type(exc).__name__}: {str(exc)[:400]}",
                )
                db.add(asset)
                db.flush()
                assets.append(asset)
                continue

            for stored in stored_images:
                asset = _create_image_localization_asset_from_source(
                    job.id,
                    stored.filename,
                    stored.content_type,
                    lambda stored_image=stored: stored_image,
                )
                db.add(asset)
                db.flush()
                assets.append(asset)

        job.status = "uploaded" if any(asset.status != "failed" for asset in assets) else "failed"
        db.commit()
        db.refresh(job)
        return {"ok": True, "job": _image_localization_job_payload(job)}
    finally:
        db.close()


def _create_image_localization_asset_from_source(job_id: int, filename: str, content_type: str, store_fn) -> ImageLocalizationAsset:
    try:
        stored = store_fn()
        prepared = prepare_asset(stored.path)
        return ImageLocalizationAsset(
            job_id=job_id,
            filename=stored.filename,
            content_type=stored.content_type,
            original_path=stored.path,
            cropped_path=prepared["cropped_path"],
            crop_json=json.dumps(prepared["crop"], ensure_ascii=False),
            ocr_json=json.dumps(prepared["ocr"], ensure_ascii=False),
            translations_json=json.dumps(prepared["translations"], ensure_ascii=False),
            approved_texts_json=json.dumps(prepared["translations"], ensure_ascii=False),
            status="uploaded",
        )
    except ValueError as exc:
        return ImageLocalizationAsset(
            job_id=job_id,
            filename=os.path.basename(filename) or filename[:120] or "remote-image",
            content_type=content_type,
            original_path="",
            status="failed",
            error_message=str(exc),
        )
    except Exception as exc:
        return ImageLocalizationAsset(
            job_id=job_id,
            filename=os.path.basename(filename) or filename[:120] or "remote-image",
            content_type=content_type,
            original_path="",
            status="failed",
            error_message=f"{type(exc).__name__}: {str(exc)[:400]}",
        )


@app.get("/api/image-localization/jobs")
def list_image_localization_jobs():
    db = SessionLocal()
    try:
        jobs = db.query(ImageLocalizationJob).order_by(ImageLocalizationJob.id.desc()).limit(25).all()
        return [_image_localization_job_payload(job, include_assets=False) for job in jobs]
    finally:
        db.close()


@app.get("/api/image-localization/jobs/{job_id}")
def get_image_localization_job(job_id: int):
    db = SessionLocal()
    try:
        job = db.query(ImageLocalizationJob).filter(ImageLocalizationJob.id == job_id).first()
        if not job:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return _image_localization_job_payload(job)
    finally:
        db.close()


@app.put("/api/image-localization/jobs/{job_id}/assets/{asset_id}/texts")
async def update_image_localization_texts(job_id: int, asset_id: int, request: Request):
    data = await request.json()
    texts = data.get("texts")
    if not isinstance(texts, list):
        return JSONResponse({"error": "texts_required"}, status_code=400)

    normalized = []
    for index, item in enumerate(texts, start=1):
        if not isinstance(item, dict):
            continue
        translated = (item.get("translated_text") or item.get("source_text") or "").strip()
        if not translated:
            continue
        normalized.append({
            "id": item.get("id") or f"text_{index}",
            "source_text": item.get("source_text") or "",
            "translated_text": translated,
            "x": int(float(item.get("x") or 0)),
            "y": int(float(item.get("y") or 0)),
            "width": max(20, int(float(item.get("width") or 220))),
            "height": max(20, int(float(item.get("height") or 80))),
            "confidence": item.get("confidence") or 0,
            "align": item.get("align") or "center",
            "auto_fit": bool(item.get("auto_fit", True)),
        })

    db = SessionLocal()
    try:
        asset = db.query(ImageLocalizationAsset).filter(
            ImageLocalizationAsset.id == asset_id,
            ImageLocalizationAsset.job_id == job_id,
        ).first()
        if not asset:
            return JSONResponse({"error": "not_found"}, status_code=404)
        if asset.cropped_path and os.path.exists(asset.cropped_path):
            with Image.open(asset.cropped_path) as image:
                normalized = fit_text_boxes_to_content(normalized, image.size, auto_only=True)
        asset.approved_texts_json = json.dumps(normalized, ensure_ascii=False)
        asset.status = "approved"
        job = db.query(ImageLocalizationJob).filter(ImageLocalizationJob.id == job_id).first()
        if job:
            _sync_image_localization_job_status(job)
        db.commit()
        if job:
            db.refresh(job)
        db.refresh(asset)
        return {"ok": True, "asset": _image_localization_asset_payload(asset), "job": _image_localization_job_payload(job) if job else None}
    finally:
        db.close()


@app.post("/api/image-localization/jobs/{job_id}/assets/{asset_id}/ocr")
async def run_image_localization_asset_ocr(job_id: int, asset_id: int, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    options = {
        "engine": body.get("engine") if isinstance(body, dict) else None,
        "ocr_model": body.get("ocr_model") if isinstance(body, dict) else None,
        "translation_model": body.get("translation_model") if isinstance(body, dict) else None,
    }
    started = time.perf_counter()
    usage_events = []
    _set_ocr_progress(
        job_id,
        asset_id,
        "queued",
        "OCR isteği alındı.",
        {
            "engine": options.get("engine") or "auto",
            "ocr_model": options.get("ocr_model") or "gemini-2.5-flash",
            "translation_model": options.get("translation_model") or "gemini-2.5-flash",
        },
    )

    def progress_callback(step: str, message: str, meta: dict | None = None) -> None:
        _set_ocr_progress(job_id, asset_id, step, message, meta)

    db = SessionLocal()
    try:
        job = db.query(ImageLocalizationJob).filter(ImageLocalizationJob.id == job_id).first()
        asset = db.query(ImageLocalizationAsset).filter(
            ImageLocalizationAsset.id == asset_id,
            ImageLocalizationAsset.job_id == job_id,
        ).first()
        if not asset:
            return JSONResponse({"error": "not_found"}, status_code=404)
        if not asset.original_path or not os.path.exists(asset.original_path):
            asset.status = "failed"
            asset.error_message = "original_image_missing"
            db.commit()
            db.refresh(asset)
            return {"ok": False, "asset": _image_localization_asset_payload(asset)}

        asset.status = "processing"
        asset.error_message = ""
        db.commit()

        with capture_llm_usage() as usage_events:
            result = await asyncio.to_thread(
                run_ocr_pipeline,
                asset.original_path,
                options,
                progress_callback,
            )
        translations = result.get("translations", [])
        asset.ocr_json = json.dumps(result.get("ocr", []), ensure_ascii=False)
        asset.translations_json = json.dumps(translations, ensure_ascii=False)
        asset.approved_texts_json = json.dumps(translations, ensure_ascii=False)
        asset.status = "needs_review"
        asset.error_message = ""
        add_llm_usage_events(
            db,
            usage_events,
            source="image-localizer",
            operation="ocr-pipeline",
            latency_ms=_elapsed_ms(started),
        )
        if job:
            _sync_image_localization_job_status(job)
        db.commit()
        if job:
            db.refresh(job)
        db.refresh(asset)
        _set_ocr_progress(job_id, asset_id, "saved", "OCR sonuçları kaydedildi.", {"count": len(translations)}, status="complete")
        return {"ok": True, "asset": _image_localization_asset_payload(asset), "job": _image_localization_job_payload(job) if job else None}
    except Exception as exc:
        asset = db.query(ImageLocalizationAsset).filter(
            ImageLocalizationAsset.id == asset_id,
            ImageLocalizationAsset.job_id == job_id,
        ).first()
        if asset:
            asset.status = "failed"
            asset.error_message = f"{type(exc).__name__}: {str(exc)[:400]}"
            _set_ocr_progress(job_id, asset_id, "failed", asset.error_message, {"error": type(exc).__name__}, status="failed")
            if usage_events:
                add_llm_usage_events(
                    db,
                    usage_events,
                    source="image-localizer",
                    operation="ocr-pipeline",
                    latency_ms=_elapsed_ms(started),
                    success=False,
                    error=exc,
                )
            db.commit()
            db.refresh(asset)
            return {"ok": False, "asset": _image_localization_asset_payload(asset)}
        return JSONResponse({"error": "ocr_failed"}, status_code=500)
    finally:
        db.close()


@app.get("/api/image-localization/jobs/{job_id}/assets/{asset_id}/ocr-progress")
def get_image_localization_asset_ocr_progress(job_id: int, asset_id: int):
    return _get_ocr_progress(job_id, asset_id)


@app.post("/api/image-localization/jobs/{job_id}/generate")
def generate_image_localization_job(job_id: int):
    db = SessionLocal()
    try:
        job = db.query(ImageLocalizationJob).filter(ImageLocalizationJob.id == job_id).first()
        if not job:
            return JSONResponse({"error": "not_found"}, status_code=404)
        job.status = "generating"
        db.commit()

        for asset in job.assets:
            _generate_image_localization_asset(asset)
        _sync_image_localization_job_status(job)
        db.commit()
        db.refresh(job)
        return {"ok": True, "job": _image_localization_job_payload(job)}
    finally:
        db.close()


@app.post("/api/image-localization/jobs/{job_id}/assets/{asset_id}/generate")
def generate_image_localization_asset(job_id: int, asset_id: int):
    db = SessionLocal()
    try:
        job = db.query(ImageLocalizationJob).filter(ImageLocalizationJob.id == job_id).first()
        if not job:
            return JSONResponse({"error": "not_found"}, status_code=404)
        asset = db.query(ImageLocalizationAsset).filter(
            ImageLocalizationAsset.id == asset_id,
            ImageLocalizationAsset.job_id == job_id,
        ).first()
        if not asset:
            return JSONResponse({"error": "not_found"}, status_code=404)
        _generate_image_localization_asset(asset)
        _sync_image_localization_job_status(job)
        db.commit()
        db.refresh(job)
        db.refresh(asset)
        return {
            "ok": True,
            "asset": _image_localization_asset_payload(asset),
            "job": _image_localization_job_payload(job),
        }
    finally:
        db.close()


def _generate_image_localization_asset(asset: ImageLocalizationAsset) -> None:
    if asset.status == "failed" and not asset.cropped_path:
        return
    try:
        texts = safe_json_loads(asset.approved_texts_json, [])
        if not asset.cropped_path or not os.path.exists(asset.cropped_path):
            raise RuntimeError("cropped_image_missing")
        asset.output_path = generate_output(asset.cropped_path, texts)
        asset.status = "complete"
        asset.error_message = ""
    except Exception as exc:
        asset.status = "failed"
        asset.error_message = f"{type(exc).__name__}: {str(exc)[:400]}"


def _sync_image_localization_job_status(job: ImageLocalizationJob) -> None:
    statuses = [asset.status for asset in job.assets]
    if not statuses:
        job.status = "created"
    elif any(status == "processing" for status in statuses):
        job.status = "processing"
    elif all(status == "complete" for status in statuses):
        job.status = "complete"
    elif any(status in {"needs_review", "approved"} for status in statuses):
        job.status = "needs_review"
    elif any(status == "uploaded" for status in statuses):
        job.status = "uploaded"
    else:
        job.status = "failed"


@app.get("/api/image-localization/assets/{asset_id}/file")
def get_image_localization_asset_file(asset_id: int, kind: str = "cropped"):
    db = SessionLocal()
    try:
        asset = db.query(ImageLocalizationAsset).filter(ImageLocalizationAsset.id == asset_id).first()
        if not asset:
            return JSONResponse({"error": "not_found"}, status_code=404)
        paths = {
            "original": asset.original_path,
            "cropped": asset.cropped_path,
            "output": asset.output_path,
        }
        path = paths.get(kind)
        if not path or not os.path.exists(path):
            return JSONResponse({"error": "file_not_found"}, status_code=404)
        media_type = "image/png" if path.lower().endswith(".png") else asset.content_type
        filename = os.path.basename(path)
        return FileResponse(path, media_type=media_type, filename=filename if kind == "output" else None)
    finally:
        db.close()


def _image_localization_job_payload(job: ImageLocalizationJob, include_assets: bool = True) -> dict:
    payload = {
        "id": job.id,
        "target_language": job.target_language,
        "status": job.status,
        "notes": job.notes or "",
        "asset_count": len(job.assets or []),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }
    if include_assets:
        payload["assets"] = [_image_localization_asset_payload(asset) for asset in sorted(job.assets, key=lambda item: item.id)]
    return payload


def _image_localization_asset_payload(asset: ImageLocalizationAsset) -> dict:
    output_version = asset.updated_at.isoformat() if asset.updated_at else str(asset.id)
    return {
        "id": asset.id,
        "job_id": asset.job_id,
        "filename": asset.filename,
        "content_type": asset.content_type,
        "status": asset.status,
        "error_message": asset.error_message or "",
        "crop": safe_json_loads(asset.crop_json, {}),
        "ocr": safe_json_loads(asset.ocr_json, []),
        "translations": safe_json_loads(asset.translations_json, []),
        "approved_texts": safe_json_loads(asset.approved_texts_json, []),
        "original_url": f"/api/image-localization/assets/{asset.id}/file?kind=original" if asset.original_path else "",
        "cropped_url": f"/api/image-localization/assets/{asset.id}/file?kind=cropped" if asset.cropped_path else "",
        "output_url": f"/api/image-localization/assets/{asset.id}/file?kind=output&v={output_version}" if asset.output_path else "",
        "created_at": asset.created_at.isoformat() if asset.created_at else None,
        "updated_at": asset.updated_at.isoformat() if asset.updated_at else None,
    }


# ============ DASHBOARD ROUTES ============

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"active": "dashboard"})


@app.get("/dashboard/conversations", response_class=HTMLResponse)
def conversations_page(request: Request):
    return templates.TemplateResponse(request, "conversations.html", {"active": "conversations"})


@app.get("/dashboard/handoffs", response_class=HTMLResponse)
def handoffs_page(request: Request):
    return templates.TemplateResponse(request, "handoffs.html", {"active": "handoffs"})


@app.get("/dashboard/agents", response_class=HTMLResponse)
def agents_page(request: Request):
    return templates.TemplateResponse(request, "agents.html", {"active": "agents"})


@app.get("/dashboard/llm", response_class=HTMLResponse)
def llm_page(request: Request):
    return templates.TemplateResponse(request, "llm.html", {"active": "llm"})


@app.get("/dashboard/channels", response_class=HTMLResponse)
def channels_page(request: Request):
    return templates.TemplateResponse(request, "channels.html", {"active": "channels"})


@app.get("/dashboard/routes", response_class=HTMLResponse)
def routes_page(request: Request):
    return templates.TemplateResponse(request, "routes.html", {"active": "routes"})


@app.get("/dashboard/knowledge", response_class=HTMLResponse)
def knowledge_page(request: Request):
    return templates.TemplateResponse(request, "knowledge.html", {"active": "knowledge"})


@app.get("/dashboard/image-localizer", response_class=HTMLResponse)
def image_localizer_page(request: Request):
    return templates.TemplateResponse(request, "image_localizer.html", {"active": "image_localizer"})
