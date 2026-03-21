import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send_message(chat_id: int | str, text: str):
    """Telegram mesajı gönder"""
    # tg_ prefix'ini çıkar
    chat_id = str(chat_id).removeprefix("tg_")
    # Uzun mesajları 4096 karaktere böl
    chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
    for chunk in chunks:
        resp = requests.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown",
        })
        if resp.status_code != 200:
            # Markdown parse hatası olursa düz text gönder
            requests.post(f"{TELEGRAM_API}/sendMessage", json={
                "chat_id": chat_id,
                "text": chunk,
            })
        else:
            print(f"Telegram mesaj gönderildi → {chat_id}")


def extract_message(payload: dict) -> tuple[str, str, str] | None:
    """Telegram webhook payload'dan chat_id, mesaj metni ve msg_id çıkar"""
    try:
        message = payload.get("message")
        if not message:
            return None

        chat_id = str(message["chat"]["id"])
        msg_id = str(message["message_id"])
        text = message.get("text", "")

        if not text:
            return None

        return chat_id, text, msg_id
    except (KeyError, IndexError):
        return None


def setup_webhook(webhook_url: str):
    """Telegram webhook'u ayarla"""
    resp = requests.post(f"{TELEGRAM_API}/setWebhook", json={
        "url": webhook_url,
    })
    print(f"Telegram webhook: {resp.json()}")
    return resp.json()
