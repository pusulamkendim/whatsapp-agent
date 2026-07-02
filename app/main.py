import json
import os
import hmac
import hashlib
import secrets
import re
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from fastapi import FastAPI, Request, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import OperationalError, ProgrammingError
from app.config import WHATSAPP_VERIFY_TOKEN
from app.database import init_db, SessionLocal
from app.whatsapp import send_message as wa_send, extract_message as wa_extract
from app.telegram import send_message as tg_send, extract_message as tg_extract, setup_webhook as tg_setup_webhook
from app.instagram import send_message as ig_send, extract_message as ig_extract
from app import retreat_agent
from app.agent_registry import run_agent
from app.llm import (
    MODEL_OPTIONS,
    capture_llm_usage,
    parse_model_ref,
    provider_label,
    record_gemini_usage,
    run_openai_simple_chat,
    summarize_llm_usage,
)
from app.models import (
    Agent,
    AgentKnowledgeBase,
    ChannelAccount,
    Conversation,
    DailyStat,
    Handoff,
    KnowledgeBase,
    KnowledgeDocument,
    LlmModel,
    LlmProvider,
    LlmUsageLog,
    Route,
)
from app.router import find_channel_account, resolve_route

app = FastAPI(title="WhatsApp Multi-Agent")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

ADMIN_SECRET = os.getenv("ADMIN_PASSWORD") or os.getenv("ADMIN_TOKEN") or ""
ADMIN_COOKIE = "agent_admin_session"
IS_PRODUCTION = bool(os.getenv("COOLIFY_RESOURCE_UUID")) or os.getenv("ENV", "").lower() == "production"

# Agent routing: müşteri hangi agent'a bağlı? Key: channel_account_id:external_user_id
customer_agents: dict[str, int] = {}

# İşlenmiş mesaj ID'leri (duplicate önleme)
processed_messages: OrderedDict[str, bool] = OrderedDict()
MAX_PROCESSED = 1000

@app.on_event("startup")
def startup():
    init_db()
    # Restoran menü verisi yükle (SQLite sıfırlanırsa diye)
    from app.models import Restaurant, MenuItem
    db = SessionLocal()
    if db.query(Restaurant).count() == 0:
        import importlib, seed_menu
        seed_menu.seed()
    db.close()
    # Telegram webhook ayarla
    import os
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


@app.post("/api/llm/models/test")
async def test_llm_model(request: Request):
    data = await request.json()
    model_ref = data.get("model_ref") or "gemini:gemini-2.5-flash"
    message = data.get("message") or "Reply with OK only."
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
        return {"ok": True, "model_ref": model_ref, "response": text, "usage": usage}
    except Exception as exc:
        return JSONResponse({"ok": False, "model_ref": model_ref, "error": str(exc)[:800]}, status_code=400)


@app.get("/api/llm/usage")
def get_llm_usage(
    limit: int = Query(20, ge=1, le=100),
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
            "by_provider": _usage_breakdown(all_rows, "provider"),
            "by_model": _usage_breakdown(all_rows, "model_ref"),
            "logs": [{
                "id": row.id,
                "agent_id": row.agent_id,
                "agent_slug": row.agent.slug if row.agent else "",
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
    return {
        "id": doc.id,
        "knowledge_base_id": doc.knowledge_base_id,
        "filename": doc.filename,
        "content_type": doc.content_type,
        "source_type": doc.source_type,
        "active": doc.active,
        "character_count": len(doc.content or ""),
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
        db.commit()
        return {"ok": True}
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
