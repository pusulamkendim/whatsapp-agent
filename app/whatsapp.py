import requests
import os
import json
from app.config import WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID

GRAPH_VERSION = "v21.0"


def _credentials(channel_account=None) -> tuple[str | None, str | None]:
    if not channel_account:
        return WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_ACCESS_TOKEN

    try:
        refs = json.loads(channel_account.credentials_json or "{}")
    except json.JSONDecodeError:
        refs = {}

    phone_number_id = refs.get("phone_number_id") or channel_account.external_id or WHATSAPP_PHONE_NUMBER_ID
    if refs.get("phone_number_id_env"):
        phone_number_id = os.getenv(refs["phone_number_id_env"], phone_number_id)

    access_token = WHATSAPP_ACCESS_TOKEN
    if refs.get("access_token_env"):
        access_token = os.getenv(refs["access_token_env"], access_token)
    elif refs.get("access_token"):
        access_token = refs["access_token"]

    return phone_number_id, access_token


def send_message(to: str, text: str, channel_account=None):
    """WhatsApp mesajı gönder"""
    phone_number_id, access_token = _credentials(channel_account)
    api_url = f"https://graph.facebook.com/{GRAPH_VERSION}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Uzun mesajları 4096 karaktere böl (WhatsApp limiti)
    chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]

    for chunk in chunks:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": chunk},
        }
        resp = requests.post(api_url, headers=headers, json=payload)
        if resp.status_code != 200:
            print(f"WhatsApp gönderim hatası: {resp.status_code} {resp.text}")
        else:
            print(f"WhatsApp mesaj gönderildi → {to}")


def extract_message(payload: dict) -> tuple[str, str, str, str] | None:
    """Webhook payload'dan müşteri numarası, mesaj metni, mesaj ID'si ve phone_number_id çıkar"""
    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        metadata = value.get("metadata", {})
        phone_number_id = metadata.get("phone_number_id", "")

        # Sadece mesaj içeren webhook'ları işle
        if "messages" not in value:
            return None

        message = value["messages"][0]
        sender = message["from"]  # müşteri telefon numarası
        msg_id = message.get("id", "")

        # Sadece text mesajları destekle (şimdilik)
        if message["type"] == "text":
            text = message["text"]["body"]
            return sender, text, msg_id, phone_number_id

        return None
    except (KeyError, IndexError):
        return None
