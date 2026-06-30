import json
import os
import hmac
import hashlib
import secrets
from collections import OrderedDict
from datetime import date
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.config import WHATSAPP_VERIFY_TOKEN
from app.database import init_db, SessionLocal
from app.whatsapp import send_message as wa_send, extract_message as wa_extract
from app.telegram import send_message as tg_send, extract_message as tg_extract, setup_webhook as tg_setup_webhook
from app.instagram import send_message as ig_send, extract_message as ig_extract
from app import retreat_agent
from app.agent_registry import run_agent
from app.models import Agent, ChannelAccount, Conversation, DailyStat, Handoff, Route
from app.router import find_channel_account, resolve_route

app = FastAPI(title="WhatsApp Multi-Agent")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

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
        tg_setup_webhook(f"{base_url}/telegram/webhook")
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

        user_key = f"{channel_account.id}:{sender}"
        if user_key in customer_agents:
            agent = db.query(Agent).filter(Agent.id == customer_agents[user_key], Agent.active == True).first()
            if agent:
                return {
                    "channel_account_id": channel_account.id,
                    "agent_id": agent.id,
                    "agent_slug": agent.slug,
                    "clean_text": text,
                }

        resolution = resolve_route(db, channel_account, text, metadata)
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
            model=data.get("model") or "gemini-2.5-flash",
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
        return {"ok": True}
    finally:
        db.close()


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


@app.get("/api/knowledge")
def get_knowledge():
    """Knowledge base içeriği"""
    kb_path = os.path.join(os.path.dirname(__file__), "..", "retreat_docs", "knowledge_base.md")
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except FileNotFoundError:
        return {"content": ""}


@app.put("/api/knowledge")
async def update_knowledge(request: Request):
    """Knowledge base güncelle"""
    data = await request.json()
    kb_path = os.path.join(os.path.dirname(__file__), "..", "retreat_docs", "knowledge_base.md")
    with open(kb_path, "w", encoding="utf-8") as f:
        f.write(data["content"])
    # Retreat agent'ın system prompt'unu yeniden yükle
    with open(kb_path, "r", encoding="utf-8") as f:
        retreat_agent.KNOWLEDGE_BASE = f.read()
    retreat_agent.SYSTEM_PROMPT = retreat_agent.build_system_prompt(retreat_agent.KNOWLEDGE_BASE)
    return {"ok": True}


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


@app.get("/dashboard/channels", response_class=HTMLResponse)
def channels_page(request: Request):
    return templates.TemplateResponse(request, "channels.html", {"active": "channels"})


@app.get("/dashboard/routes", response_class=HTMLResponse)
def routes_page(request: Request):
    return templates.TemplateResponse(request, "routes.html", {"active": "routes"})


@app.get("/dashboard/knowledge", response_class=HTMLResponse)
def knowledge_page(request: Request):
    return templates.TemplateResponse(request, "knowledge.html", {"active": "knowledge"})
