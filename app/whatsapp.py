import requests
from app.config import WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID

API_URL = f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
HEADERS = {
    "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
    "Content-Type": "application/json",
}


def send_message(to: str, text: str):
    """WhatsApp mesajı gönder"""
    # Uzun mesajları 4096 karaktere böl (WhatsApp limiti)
    chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]

    for chunk in chunks:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": chunk},
        }
        resp = requests.post(API_URL, headers=HEADERS, json=payload)
        if resp.status_code != 200:
            print(f"WhatsApp gönderim hatası: {resp.status_code} {resp.text}")
        else:
            print(f"WhatsApp mesaj gönderildi → {to}")


def extract_message(payload: dict) -> tuple[str, str, str] | None:
    """Webhook payload'dan müşteri numarası, mesaj metni ve mesaj ID'sini çıkar"""
    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # Sadece mesaj içeren webhook'ları işle
        if "messages" not in value:
            return None

        message = value["messages"][0]
        sender = message["from"]  # müşteri telefon numarası
        msg_id = message.get("id", "")

        # Sadece text mesajları destekle (şimdilik)
        if message["type"] == "text":
            text = message["text"]["body"]
            return sender, text, msg_id

        return None
    except (KeyError, IndexError):
        return None
