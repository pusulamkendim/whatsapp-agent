import json
from collections import OrderedDict
from datetime import date
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from app.config import WHATSAPP_VERIFY_TOKEN
from app.database import init_db, SessionLocal
from app.whatsapp import send_message, extract_message
from app.agent import chat as restaurant_chat
from app.retreat_agent import chat as retreat_chat
from app.models import Conversation, DailyStat

app = FastAPI(title="WhatsApp Multi-Agent")

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
    print("✅ WhatsApp Multi-Agent başlatıldı!")


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


def process_message(sender: str, clean_text: str, agent_type: str, msg_id: str = ""):
    """Mesajı arka planda işle"""
    # Kullanıcı mesajını kaydet
    save_message(sender, agent_type, "user", clean_text, msg_id)

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

        print(f"🤖 [{agent_type}] Cevap: {response}")
        send_message(sender, response)
    except Exception as e:
        print(f"❌ Hata: {e}")
        send_message(sender, "Bir sorun oluştu, lütfen tekrar deneyin.")


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Gelen WhatsApp mesajlarını işle"""
    payload = await request.json()

    # Raw payload logla (debug)
    print(f"📦 Webhook payload: {json.dumps(payload, ensure_ascii=False)[:500]}")

    result = extract_message(payload)
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
    background_tasks.add_task(process_message, sender, clean_text, agent_type, msg_id)

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
    from app.models import Handoff
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
