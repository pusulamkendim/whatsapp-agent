from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from app.config import WHATSAPP_VERIFY_TOKEN
from app.database import init_db, SessionLocal
from app.whatsapp import send_message, extract_message
from app.agent import chat as restaurant_chat
from app.retreat_agent import chat as retreat_chat

app = FastAPI(title="WhatsApp Multi-Agent")

# Agent routing: müşteri hangi agent'a bağlı?
# İlk mesaja göre belirlenir, sonra hafızada kalır
customer_agents: dict[str, str] = {}

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
    print("✅ WhatsApp Multi-Agent başlatıldı!")
    print("   Aktif agent'lar: restaurant, retreat")


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


@app.post("/webhook")
async def handle_webhook(request: Request):
    """Gelen WhatsApp mesajlarını işle"""
    payload = await request.json()

    result = extract_message(payload)
    if not result:
        return {"status": "ignored"}

    sender, text = result
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

        print(f"🤖 [{agent_type}] Cevap: {response[:100]}...")
        send_message(sender, response)
    except Exception as e:
        print(f"❌ Hata: {e}")
        send_message(sender, "Bir sorun oluştu, lütfen tekrar deneyin.")

    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "status": "running",
        "agents": ["restaurant", "retreat"],
        "routing": dict(AGENT_CODES),
    }
