import os
import requests

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")


def send_message(recipient_id: str, text: str):
    """Instagram DM gönder"""
    # ig_ prefix'ini çıkar
    recipient_id = str(recipient_id).removeprefix("ig_")

    url = f"https://graph.facebook.com/v22.0/me/messages"
    headers = {
        "Authorization": f"Bearer {INSTAGRAM_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Uzun mesajları 1000 karaktere böl (Instagram limiti daha kısa)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    for chunk in chunks:
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": chunk},
        }
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            print(f"Instagram gönderim hatası: {resp.status_code} {resp.text}")
        else:
            print(f"Instagram mesaj gönderildi → {recipient_id}")


def extract_message(payload: dict) -> tuple[str, str, str, bool] | None:
    """Instagram webhook payload'dan sender_id, mesaj, msg_id ve reklamdan mı geldiğini çıkar"""
    try:
        if payload.get("object") != "instagram":
            return None

        entry = payload["entry"][0]
        messaging = entry["messaging"][0]

        sender_id = messaging["sender"]["id"]
        msg_id = ""
        text = ""
        is_from_ad = False

        # Mesaj var mı?
        if "message" in messaging:
            msg_id = messaging["message"].get("mid", "")
            text = messaging["message"].get("text", "")

        # Referral (reklamdan gelen) kontrol
        if "referral" in messaging:
            referral = messaging["referral"]
            if referral.get("source") == "ADS":
                is_from_ad = True

        if not text:
            return None

        return sender_id, text, msg_id, is_from_ad
    except (KeyError, IndexError):
        return None
