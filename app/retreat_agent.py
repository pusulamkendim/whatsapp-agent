from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY
from app.whatsapp import send_message
from app.llm import (
    GEMINI_CLIENT,
    gemini_tool_from_openai_tools,
    is_gemini_model,
    parse_model_ref,
    record_gemini_usage,
    run_openai_tool_loop,
)
import os
import json

client = GEMINI_CLIENT

# İnziva sahiplerinin WhatsApp numaraları (virgülle ayrılmış)
OWNER_PHONES = [p.strip() for p in os.getenv("RETREAT_OWNER_PHONES", "905555574128").split(",")]

# Knowledge base'i yükle
KB_PATH = os.path.join(os.path.dirname(__file__), "..", "retreat_docs", "knowledge_base.md")
with open(KB_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = f.read()

tool_definitions_openai = [
    {
        "type": "function",
        "function": {
            "name": "handoff_to_human",
            "description": "Müşteriyi inziva ekibiyle görüştürmek için kullan. Müşteri kayıt olmak istediğinde, özel sorular sorduğunda, ödeme/taksit detayları istediğinde veya sağlık/psikolojik durum paylaştığında çağır. Adını mutlaka sor ve özet bilgi ile birlikte gönder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string", "description": "Müşterinin adı soyadı"},
                    "conversation_summary": {"type": "string", "description": "Konuşmanın kısa özeti: ne sordu, neyle ilgilendi, özel durumu var mı"},
                    "interest_level": {"type": "string", "description": "İlgi seviyesi: yüksek, orta, düşük"},
                },
                "required": ["customer_name", "conversation_summary"],
            },
        },
    },
]

tool_definitions = gemini_tool_from_openai_tools(tool_definitions_openai)

def build_system_prompt(knowledge_base: str) -> str:
    return f"""Sen Samma Karuna'nın Koh Phangan Nefes ve Tantra İnzivası (23-30 Mayıs 2026) hakkında bilgi veren WhatsApp asistanısın.

Görevin:
- İnziva hakkında soruları samimi, sıcak ve bilgilendirici şekilde yanıtlamak
- Merak uyandırmak ve ilham vermek (ama baskıcı satış yapmamak)
- Kişinin sorularını dinleyip, durumuna uygun bilgi vermek
- Gerektiğinde handoff_to_human tool'unu kullanarak insana yönlendirmek

Tarzın:
- Türkçe, samimi, sıcak ama profesyonel
- Spiritüel jargonu fazla kullanma, anlaşılır ol
- Kısa ve öz cevaplar ver, uzun paragraflar yazma
- Emoji kullanabilirsin ama abartma

Kurallar:
- SADECE aşağıdaki bilgi bankasındaki bilgileri kullan, bilmediğin şeyleri UYDURMA
- Fiyatları, tarihleri, program detaylarını bilgi bankasından al
- Eğer bilgi bankasında olmayan bir soru gelirse "Bu konuda sizi doğrudan ekibimizle görüştürmek isterim" de
- İnziva dışında konularla ilgili sorulara yanıt verme, nazikçe konuyu inzivaya getir

handoff_to_human tool'unu şu durumlarda kullan:
- Müşteri "kayıt olmak istiyorum", "katılmak istiyorum" dediğinde
- Kişisel sağlık/psikolojik durumlarla ilgili sorular geldiğinde
- Ödeme detayları, taksit, özel indirim talep edildiğinde
- Bilgi bankasında olmayan bir soru geldiğinde ve sen cevaplayamıyorsan
- Müşteri ısrarla bir konuda detay istiyorsa ve bilgi bankasında yoksa
- 4-5 mesajdan sonra ilgi devam ediyorsa

Tool'u çağırmadan ÖNCE mutlaka müşterinin adını sor. Adını aldıktan sonra tool'u çağır.
Eğer müşteri adını vermek istemezse, "İlgilenen kişi" olarak yine tool'u çağır, adı zorunlu tutma.
Tool çağrıldıktan sonra müşteriye "Bilgilerinizi ekibimize ilettim, Özgür Can en kısa sürede sizinle iletişime geçecek 🙏" de.

---

BİLGİ BANKASI:

{knowledge_base}
"""


SYSTEM_PROMPT = build_system_prompt(KNOWLEDGE_BASE)

# Konuşma geçmişi
conversations: dict[str, list] = {}
openai_conversations: dict[str, list[dict]] = {}
# Handoff yapılmış mı takibi
handoffs: dict[str, bool] = {}


def execute_handoff(customer_phone: str, customer_name: str, summary: str, interest: str):
    """Özgür Can'a bildirim mesajı gönder + DB'ye kaydet"""
    from app.database import SessionLocal
    from app.models import Handoff

    # DB'ye kaydet
    db = SessionLocal()
    try:
        db.add(Handoff(
            customer_phone=customer_phone,
            customer_name=customer_name,
            conversation_summary=summary,
            interest_level=interest or "belirtilmedi",
        ))
        db.commit()
    except Exception as e:
        print(f"⚠️ Handoff DB kayıt hatası: {e}")
    finally:
        db.close()

    # WhatsApp bildirimi
    message = (
        f"📋 Yeni İnziva İlgilisi\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"👤 İsim: {customer_name}\n"
        f"📱 Telefon: +{customer_phone}\n"
        f"🔥 İlgi: {interest or 'belirtilmedi'}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💬 Özet:\n{summary}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"wa.me/{customer_phone}"
    )
    for phone in OWNER_PHONES:
        send_message(phone, message)
    return "Bilgiler ekibe iletildi."


def chat(
    customer_id: str,
    message: str,
    knowledge_base: str | None = None,
    system_prompt: str | None = None,
    model: str = "gemini:gemini-2.5-flash",
) -> str:
    """İnziva sorusunu yanıtla"""
    if system_prompt:
        prompt = system_prompt
    elif knowledge_base:
        prompt = build_system_prompt(knowledge_base)
    else:
        prompt = SYSTEM_PROMPT

    if not is_gemini_model(model):
        return _chat_openai_compatible(customer_id, message, prompt, model)

    if customer_id not in conversations:
        conversations[customer_id] = []

    history = conversations[customer_id]

    history.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    ))

    response = client.models.generate_content(
        model=parse_model_ref(model)[1],
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            tools=[tool_definitions],
            temperature=0.7,
        ),
    )
    record_gemini_usage(model, response)

    # Tool call loop
    max_iterations = 3
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_calls = [p for p in parts if p.function_call]

        if not function_calls:
            break

        history.append(candidate.content)

        function_responses = []
        for part in function_calls:
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}

            if fc.name == "handoff_to_human":
                result = execute_handoff(
                    customer_phone=customer_id,
                    customer_name=args.get("customer_name", "Bilinmiyor"),
                    summary=args.get("conversation_summary", ""),
                    interest=args.get("interest_level", ""),
                )
                handoffs[customer_id] = True
            else:
                result = "Bilinmeyen tool"

            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                )
            )

        history.append(types.Content(
            role="user",
            parts=function_responses,
        ))

        response = client.models.generate_content(
            model=parse_model_ref(model)[1],
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                tools=[tool_definitions],
                temperature=0.7,
            ),
        )
        record_gemini_usage(model, response)

    final_text = response.candidates[0].content.parts[0].text
    history.append(response.candidates[0].content)

    if len(history) > 40:
        conversations[customer_id] = history[-40:]

    return final_text


def _chat_openai_compatible(customer_id: str, message: str, prompt: str, model: str) -> str:
    if customer_id not in openai_conversations:
        openai_conversations[customer_id] = [{"role": "system", "content": prompt}]

    history = openai_conversations[customer_id]
    history.append({"role": "user", "content": message})

    def call_tool(tool_name: str, args: dict) -> str:
        if tool_name != "handoff_to_human":
            return "Bilinmeyen tool"
        result = execute_handoff(
            customer_phone=customer_id,
            customer_name=args.get("customer_name", "Bilinmiyor"),
            summary=args.get("conversation_summary", ""),
            interest=args.get("interest_level", ""),
        )
        handoffs[customer_id] = True
        return result

    final_text = run_openai_tool_loop(
        model,
        history,
        tool_definitions_openai,
        call_tool,
        max_iterations=3,
    )

    if len(history) > 42:
        openai_conversations[customer_id] = [history[0], *history[-41:]]

    return final_text
