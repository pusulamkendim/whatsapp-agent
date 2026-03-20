from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY
import os

client = genai.Client(api_key=GEMINI_API_KEY)

# Knowledge base'i yükle
KB_PATH = os.path.join(os.path.dirname(__file__), "..", "retreat_docs", "knowledge_base.md")
with open(KB_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = f.read()

SYSTEM_PROMPT = f"""Sen Samma Karuna'nın Koh Phangan Nefes ve Tantra İnzivası (23-30 Mayıs 2026) hakkında bilgi veren WhatsApp asistanısın.

Görevin:
- İnziva hakkında soruları samimi, sıcak ve bilgilendirici şekilde yanıtlamak
- Merak uyandırmak ve ilham vermek (ama baskıcı satış yapmamak)
- Kişinin sorularını dinleyip, durumuna uygun bilgi vermek
- Gerektiğinde insana yönlendirmek

Tarzın:
- Türkçe, samimi, sıcak ama profesyonel
- Spiritüel jargonu fazla kullanma, anlaşılır ol
- Kısa ve öz cevaplar ver, uzun paragraflar yazma
- Emoji kullanabilirsin ama abartma

Kurallar:
- SADECE aşağıdaki bilgi bankasındaki bilgileri kullan, bilmediğin şeyleri UYDURMA
- Fiyatları, tarihleri, program detaylarını bilgi bankasından al
- Eğer bilgi bankasında olmayan bir soru gelirse "Bu konuda sizi doğrudan ekibimizle görüştürmek isterim" de
- Kişi kayıt olmak istediğinde, detaylı sorular sorduğunda veya özel durumunu paylaştığında → insana yönlendir
- İnziva dışında konularla ilgili sorulara yanıt verme, nazikçe konuyu inzivaya getir

İnsana yönlendirme zamanı:
- "Kayıt olmak istiyorum" → "Harika! Sizinle kısa bir online görüşme planlamamız gerekiyor. Sizi Özgür Can ile görüştüreyim."
- Kişisel sağlık/psikolojik durumlarla ilgili sorular → insana yönlendir
- Ödeme detayları, taksit, özel indirim talepleri → insana yönlendir
- 3-4 mesajdan sonra ilgi devam ediyorsa → "İsterseniz sizi ekibimizle görüştüreyim, tüm sorularınızı detaylıca konuşabilirsiniz"

Yönlendirme mesajı:
"Sizi ekibimizle görüştürmek istiyorum. Özgür Can size kısa bir online görüşme için ulaşacak. Adınızı öğrenebilir miyim?"

---

BİLGİ BANKASI:

{KNOWLEDGE_BASE}
"""

# Konuşma geçmişi
conversations: dict[str, list] = {}
# İnsana yönlendirme sayacı
message_counts: dict[str, int] = {}


def chat(customer_id: str, message: str) -> str:
    """İnziva sorusunu yanıtla"""

    if customer_id not in conversations:
        conversations[customer_id] = []
        message_counts[customer_id] = 0

    message_counts[customer_id] += 1
    history = conversations[customer_id]

    # Kullanıcı mesajını ekle
    history.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    ))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )

    reply = response.candidates[0].content.parts[0].text
    history.append(response.candidates[0].content)

    # Geçmişi kırp
    if len(history) > 40:
        conversations[customer_id] = history[-40:]

    return reply
