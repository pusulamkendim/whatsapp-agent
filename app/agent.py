from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from app.config import GEMINI_API_KEY
from app.tools import execute_tool

client = genai.Client(api_key=GEMINI_API_KEY)

# Tool tanımları (Gemini function calling formatı)
tool_definitions = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_menu",
            description="Restoranın tam menüsünü kategorilere göre getirir. Müşteri menüyü görmek istediğinde veya ne yemek yiyeceğine karar veremediğinde kullan.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="search_menu",
            description="Menüde arama yapar. Müşteri belirli bir yemek, kategori veya tercih belirttiğinde kullan. Örn: 'acılı', 'vejetaryen', 'pide', 'salata'",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="Aranacak kelime veya tercih"),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="add_to_cart",
            description="Sepete ürün ekler. Müşteri bir ürün sipariş etmek istediğinde kullan.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "item_id": types.Schema(type=types.Type.INTEGER, description="Menüdeki ürün ID'si (#numara)"),
                    "quantity": types.Schema(type=types.Type.INTEGER, description="Adet (varsayılan 1)"),
                    "note": types.Schema(type=types.Type.STRING, description="Müşteri notu: az acılı, extra peynir vb."),
                },
                required=["item_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="remove_from_cart",
            description="Sepetten ürün çıkarır.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "item_id": types.Schema(type=types.Type.INTEGER, description="Çıkarılacak ürün ID'si"),
                },
                required=["item_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_cart",
            description="Mevcut sepeti ve toplamı gösterir. Müşteri sepetini görmek istediğinde veya sipariş vermeden önce kullan.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="confirm_order",
            description="Siparişi onaylar ve işletmeye gönderir. SADECE müşteri onay verdiğinde ve adres bilgisi alındığında kullan.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "delivery_address": types.Schema(type=types.Type.STRING, description="Teslimat adresi"),
                    "payment_method": types.Schema(type=types.Type.STRING, description="Ödeme yöntemi: cash veya card_on_delivery"),
                },
                required=["delivery_address"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_order_status",
            description="Mevcut siparişin durumunu sorgular.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "order_id": types.Schema(type=types.Type.INTEGER, description="Sipariş numarası"),
                },
                required=["order_id"],
            ),
        ),
    ]
)


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


def chat(
    customer_id: str,
    message: str,
    restaurant_id: int,
    restaurant_name: str,
    db: Session,
) -> str:
    """Müşteri mesajını işle ve cevap döndür"""

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
        model="gemini-2.5-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=get_system_prompt(restaurant_name),
            tools=[tool_definitions],
            temperature=0.7,
        ),
    )

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
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=get_system_prompt(restaurant_name),
                tools=[tool_definitions],
                temperature=0.7,
            ),
        )

    # Son cevabı geçmişe ekle
    final_text = response.candidates[0].content.parts[0].text
    history.append(response.candidates[0].content)

    # Geçmişi çok uzamasın diye kırp (son 30 mesaj)
    if len(history) > 30:
        conversations[customer_id] = history[-30:]

    return final_text
