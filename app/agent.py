from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from app.config import GEMINI_API_KEY
from app.tools import execute_tool
from app.llm import (
    GEMINI_CLIENT,
    gemini_tool_from_openai_tools,
    is_gemini_model,
    parse_model_ref,
    record_gemini_usage,
    run_openai_tool_loop,
)

client = GEMINI_CLIENT

tool_definitions_openai = [
    {
        "type": "function",
        "function": {
            "name": "get_menu",
            "description": "Restoranın tam menüsünü kategorilere göre getirir. Müşteri menüyü görmek istediğinde veya ne yemek yiyeceğine karar veremediğinde kullan.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "Menüde arama yapar. Müşteri belirli bir yemek, kategori veya tercih belirttiğinde kullan. Örn: 'acılı', 'vejetaryen', 'pide', 'salata'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Aranacak kelime veya tercih"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Sepete ürün ekler. Müşteri bir ürün sipariş etmek istediğinde kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer", "description": "Menüdeki ürün ID'si (#numara)"},
                    "quantity": {"type": "integer", "description": "Adet (varsayılan 1)"},
                    "note": {"type": "string", "description": "Müşteri notu: az acılı, extra peynir vb."},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": "Sepetten ürün çıkarır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer", "description": "Çıkarılacak ürün ID'si"},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "Mevcut sepeti ve toplamı gösterir. Müşteri sepetini görmek istediğinde veya sipariş vermeden önce kullan.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": "Siparişi onaylar ve işletmeye gönderir. SADECE müşteri onay verdiğinde ve adres bilgisi alındığında kullan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delivery_address": {"type": "string", "description": "Teslimat adresi"},
                    "payment_method": {"type": "string", "description": "Ödeme yöntemi: cash veya card_on_delivery"},
                },
                "required": ["delivery_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Mevcut siparişin durumunu sorgular.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "Sipariş numarası"},
                },
                "required": ["order_id"],
            },
        },
    },
]

tool_definitions = gemini_tool_from_openai_tools(tool_definitions_openai)


def get_system_prompt(restaurant_name: str) -> str:
    return f"""Sen {restaurant_name} restoranının WhatsApp sipariş asistanısın.
Müşteriyle Türkçe, samimi ve kısa konuşursun. Emoji kullanabilirsin.
Menüdeki ürünleri önerirsin, damak tadına göre yönlendirirsin.

Kurallar:
- Fiyatları SADECE get_menu/search_menu tool'undan al, ASLA uydurma
- Menüde olmayan ürünü önerme
- Her ürünün #ID numarasını kullanarak sepete ekle
- Sipariş toplamını her zaman göster
- Adres almadan siparişi onaylama
- Müşteri açıkça onaylamadan siparişi gönderme
- Müşteri selamlaşırsa kısaca karşıla ve menüyü görmek isteyip istemediğini sor
- Müşteri bir tercih belirtirse (acılı, hafif, vejetaryen vb.) search_menu ile ara ve öner"""


# Konuşma geçmişi (memory-based, MVP)
conversations: dict[str, list] = {}
openai_conversations: dict[str, list[dict]] = {}


def chat(
    customer_id: str,
    message: str,
    restaurant_id: int,
    restaurant_name: str,
    db: Session,
    model: str = "gemini:gemini-2.5-flash",
) -> str:
    """Müşteri mesajını işle ve cevap döndür"""
    if not is_gemini_model(model):
        return _chat_openai_compatible(customer_id, message, restaurant_id, restaurant_name, db, model)

    # Konuşma geçmişini al veya oluştur
    if customer_id not in conversations:
        conversations[customer_id] = []

    history = conversations[customer_id]

    # Kullanıcı mesajını ekle
    history.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    ))

    # Gemini'a gönder
    response = client.models.generate_content(
        model=parse_model_ref(model)[1],
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=get_system_prompt(restaurant_name),
            tools=[tool_definitions],
            temperature=0.7,
        ),
    )
    record_gemini_usage(model, response)

    # Tool call loop — Gemini tool çağırabilir, sonucu geri gönderip tekrar cevap alırız
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Tool call var mı kontrol et
        function_calls = [p for p in parts if p.function_call]

        if not function_calls:
            # Tool call yok, text cevap var
            break

        # Model cevabını geçmişe ekle
        history.append(candidate.content)

        # Her tool call'u çalıştır
        function_responses = []
        for part in function_calls:
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}
            result = execute_tool(fc.name, args, customer_id, db, restaurant_id)
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                )
            )

        # Tool sonuçlarını geçmişe ekle
        history.append(types.Content(
            role="user",
            parts=function_responses,
        ))

        # Tekrar Gemini'a gönder
        response = client.models.generate_content(
            model=parse_model_ref(model)[1],
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=get_system_prompt(restaurant_name),
                tools=[tool_definitions],
                temperature=0.7,
            ),
        )
        record_gemini_usage(model, response)

    # Son cevabı geçmişe ekle
    final_text = response.candidates[0].content.parts[0].text
    history.append(response.candidates[0].content)

    # Geçmişi çok uzamasın diye kırp (son 30 mesaj)
    if len(history) > 30:
        conversations[customer_id] = history[-30:]

    return final_text


def _chat_openai_compatible(
    customer_id: str,
    message: str,
    restaurant_id: int,
    restaurant_name: str,
    db: Session,
    model: str,
) -> str:
    if customer_id not in openai_conversations:
        openai_conversations[customer_id] = [
            {"role": "system", "content": get_system_prompt(restaurant_name)}
        ]

    history = openai_conversations[customer_id]
    history.append({"role": "user", "content": message})

    def call_tool(tool_name: str, args: dict) -> str:
        return execute_tool(tool_name, args, customer_id, db, restaurant_id)

    final_text = run_openai_tool_loop(
        model,
        history,
        tool_definitions_openai,
        call_tool,
        max_iterations=5,
    )

    if len(history) > 32:
        openai_conversations[customer_id] = [history[0], *history[-31:]]

    return final_text
