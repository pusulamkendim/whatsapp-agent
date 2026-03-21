import json
import os
from collections import OrderedDict
from datetime import date
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from app.config import WHATSAPP_VERIFY_TOKEN
from app.database import init_db, SessionLocal
from app.whatsapp import send_message as wa_send, extract_message as wa_extract
from app.telegram import send_message as tg_send, extract_message as tg_extract, setup_webhook as tg_setup_webhook
from app.instagram import send_message as ig_send, extract_message as ig_extract
from app.agent import chat as restaurant_chat
from app import retreat_agent
from app.retreat_agent import chat as retreat_chat
from app.models import Conversation, DailyStat, Handoff

app = FastAPI(title="WhatsApp Multi-Agent")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Agent config (runtime'da degistirilebilir)
agent_configs = {
    "retreat": {"name": "Inziva Agent", "active": True, "model": "gemini-2.5-flash"},
    "restaurant": {"name": "Restoran Agent", "active": True, "model": "gemini-2.5-flash"},
}

# Agent routing: müşteri hangi agent'a bağlı?
customer_agents: dict[str, str] = {}

# İşlenmiş mesaj ID'leri (duplicate önleme)
processed_messages: OrderedDict[str, bool] = OrderedDict()
MAX_PROCESSED = 1000

# Agent kodları (click-to-chat linki ile gelen ilk mesaj)
AGENT_CODES = {
    "LEZZET": "restaurant",
    "INZIVA": "retreat",
    "RETREAT": "retreat",
    "SAMMA": "retreat",
}

# Restoran ayarları
RESTAURANT_ID = 1
RESTAURANT_NAME = "Lezzet Durağı"


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
    print("✅ Multi-Agent başlatıldı! (WhatsApp + Telegram)")


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


def detect_agent(sender: str, text: str) -> str:
    """Mesajdan agent tipini belirle"""
    # Zaten bir agent'a atanmış mı?
    if sender in customer_agents:
        return customer_agents[sender]

    # İlk mesajdaki koda bak
    first_word = text.strip().upper().split()[0] if text.strip() else ""
    for code, agent_type in AGENT_CODES.items():
        if first_word.startswith(code):
            customer_agents[sender] = agent_type
            return agent_type

    # Varsayılan: retreat (şu anki aktif kampanya)
    customer_agents[sender] = "retreat"
    return "retreat"


# Bugün yazanları takip (unique user + new user kontrolü)
daily_seen_users: dict[str, set] = {}  # date_str -> set of phones
all_known_users: set = set()


def save_message(sender: str, agent_type: str, role: str, message: str, msg_id: str = ""):
    """Mesajı DB'ye kaydet + günlük istatistik güncelle"""
    db = SessionLocal()
    try:
        # Konuşmayı kaydet
        db.add(Conversation(
            customer_phone=sender,
            agent_type=agent_type,
            role=role,
            message=message,
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


def process_message(sender: str, clean_text: str, agent_type: str, msg_id: str = "", channel: str = "whatsapp"):
    """Mesajı arka planda işle"""
    # Kullanıcı mesajını kaydet
    save_message(sender, agent_type, "user", clean_text, msg_id)

    # Kanal bazlı mesaj gönderme
    send_fns = {"whatsapp": wa_send, "telegram": tg_send, "instagram": ig_send}
    send_fn = send_fns.get(channel, wa_send)

    try:
        if agent_type == "restaurant":
            db = SessionLocal()
            try:
                response = restaurant_chat(sender, clean_text, RESTAURANT_ID, RESTAURANT_NAME, db)
            finally:
                db.close()
        elif agent_type == "retreat":
            response = retreat_chat(sender, clean_text)
        else:
            response = "Bir sorun oluştu."

        # Agent cevabını kaydet
        save_message(sender, agent_type, "agent", response)

        print(f"🤖 [{agent_type}/{channel}] Cevap: {response}")
        send_fn(sender, response)
    except Exception as e:
        print(f"❌ Hata: {e}")
        send_fn(sender, "Bir sorun oluştu, lütfen tekrar deneyin.")


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Gelen WhatsApp mesajlarını işle"""
    payload = await request.json()

    # Raw payload logla (debug)
    print(f"📦 Webhook payload: {json.dumps(payload, ensure_ascii=False)[:500]}")

    result = wa_extract(payload)
    if not result:
        return {"status": "ignored"}

    sender, text, msg_id = result

    # Kendi business numaramızdan gelen mesajları işleme
    if sender == "905428078429":
        print(f"⏭️ Kendi numaramızdan mesaj, atlanıyor")
        return {"status": "self_message"}

    # Duplicate mesaj kontrolü (memory + DB)
    if msg_id in processed_messages:
        print(f"⏭️ Duplicate mesaj atlandı (memory): {msg_id}")
        return {"status": "duplicate"}

    # DB'den de kontrol et (restart sonrası memory boş olabilir)
    if msg_id:
        db_check = SessionLocal()
        try:
            existing = db_check.query(Conversation).filter(Conversation.msg_id == msg_id).first()
            if existing:
                print(f"⏭️ Duplicate mesaj atlandı (DB): {msg_id}")
                processed_messages[msg_id] = True
                return {"status": "duplicate"}
        finally:
            db_check.close()

    processed_messages[msg_id] = True
    if len(processed_messages) > MAX_PROCESSED:
        processed_messages.popitem(last=False)

    agent_type = detect_agent(sender, text)

    # Agent kodunu mesajdan çıkar (ilk mesajda)
    clean_text = text
    first_word = text.strip().upper().split()[0] if text.strip() else ""
    for code in AGENT_CODES:
        if first_word.startswith(code):
            clean_text = text[len(first_word):].strip()
            if not clean_text:
                clean_text = "Merhaba"
            break

    print(f"📩 [{agent_type}] {sender} → {clean_text}")

    # Hemen 200 dön, mesajı arka planda işle (Meta timeout'a takılmasın)
    background_tasks.add_task(process_message, sender, clean_text, agent_type, msg_id, "whatsapp")

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

    # Duplicate kontrolü
    if msg_id in processed_messages:
        return {"status": "duplicate"}

    if msg_id:
        db_check = SessionLocal()
        try:
            existing = db_check.query(Conversation).filter(Conversation.msg_id == msg_id).first()
            if existing:
                processed_messages[msg_id] = True
                return {"status": "duplicate"}
        finally:
            db_check.close()

    processed_messages[msg_id] = True
    if len(processed_messages) > MAX_PROCESSED:
        processed_messages.popitem(last=False)

    agent_type = detect_agent(f"tg_{chat_id}", text)

    clean_text = text
    first_word = text.strip().upper().split()[0] if text.strip() else ""
    for code in AGENT_CODES:
        if first_word.startswith(code):
            clean_text = text[len(first_word):].strip()
            if not clean_text:
                clean_text = "Merhaba"
            break

    print(f"📩 [TG/{agent_type}] {chat_id} → {clean_text}")

    background_tasks.add_task(process_message, f"tg_{chat_id}", clean_text, agent_type, msg_id, "telegram")

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

    sender_id, text, msg_id, is_from_ad = result
    msg_id = f"ig_{msg_id}"

    # Sadece reklamdan gelen mesajları işle
    # İlk mesaj reklamdan geldiyse, o kullanıcıyı "ad_users" olarak kaydet
    # Sonraki mesajları da işle (aynı konuşma devam ediyor)
    ad_user_key = f"ig_{sender_id}"

    if is_from_ad:
        customer_agents[ad_user_key] = "retreat"  # Reklamdan gelen → retreat agent
        print(f"📢 Instagram reklamdan mesaj: {sender_id}")
    elif ad_user_key not in customer_agents:
        # Reklamdan gelmeyen ve daha önce reklamdan da gelmemiş → atla
        print(f"⏭️ Instagram organik mesaj atlandı: {sender_id}")
        return {"status": "not_from_ad"}

    # Duplicate kontrolü
    if msg_id in processed_messages:
        return {"status": "duplicate"}

    if msg_id:
        db_check = SessionLocal()
        try:
            existing = db_check.query(Conversation).filter(Conversation.msg_id == msg_id).first()
            if existing:
                processed_messages[msg_id] = True
                return {"status": "duplicate"}
        finally:
            db_check.close()

    processed_messages[msg_id] = True
    if len(processed_messages) > MAX_PROCESSED:
        processed_messages.popitem(last=False)

    print(f"📩 [IG/retreat] {sender_id} → {text}")

    background_tasks.add_task(process_message, ad_user_key, text, "retreat", msg_id, "instagram")

    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "status": "running",
        "agents": ["restaurant", "retreat"],
        "routing": dict(AGENT_CODES),
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
    result = []
    for agent_type, config in agent_configs.items():
        prompt = ""
        if agent_type == "retreat":
            prompt = retreat_agent.SYSTEM_PROMPT[:2000]
        elif agent_type == "restaurant":
            from app.agent import get_system_prompt
            prompt = get_system_prompt("Lezzet Durağı")
        result.append({
            "type": agent_type,
            "name": config["name"],
            "active": config["active"],
            "model": config["model"],
            "system_prompt": prompt,
        })
    return result


@app.put("/api/agents/{agent_type}/config")
async def update_agent_config(agent_type: str, request: Request):
    """Agent ayarlarını güncelle"""
    if agent_type not in agent_configs:
        return {"error": "unknown agent"}
    data = await request.json()
    if "active" in data:
        agent_configs[agent_type]["active"] = data["active"]
    if "system_prompt" in data and agent_type == "retreat":
        retreat_agent.SYSTEM_PROMPT = data["system_prompt"]
    return {"ok": True}


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
    retreat_agent.SYSTEM_PROMPT = retreat_agent.SYSTEM_PROMPT  # trigger reload if needed
    return {"ok": True}


# ============ DASHBOARD ROUTES ============

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "active": "dashboard"})


@app.get("/dashboard/conversations", response_class=HTMLResponse)
def conversations_page(request: Request):
    return templates.TemplateResponse("conversations.html", {"request": request, "active": "conversations"})


@app.get("/dashboard/handoffs", response_class=HTMLResponse)
def handoffs_page(request: Request):
    return templates.TemplateResponse("handoffs.html", {"request": request, "active": "handoffs"})


@app.get("/dashboard/agents", response_class=HTMLResponse)
def agents_page(request: Request):
    return templates.TemplateResponse("agents.html", {"request": request, "active": "agents"})


@app.get("/dashboard/knowledge", response_class=HTMLResponse)
def knowledge_page(request: Request):
    return templates.TemplateResponse("knowledge.html", {"request": request, "active": "knowledge"})
