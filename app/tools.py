import json
from sqlalchemy.orm import Session
from app.models import MenuItem, Order


# Her müşterinin sepeti (memory-based, MVP için yeterli)
carts: dict[str, list[dict]] = {}


def get_menu(db: Session, restaurant_id: int) -> str:
    items = db.query(MenuItem).filter(
        MenuItem.restaurant_id == restaurant_id,
        MenuItem.is_available == True,
    ).order_by(MenuItem.category, MenuItem.name).all()

    if not items:
        return "Menü bulunamadı."

    menu = {}
    for item in items:
        cat = item.category
        if cat not in menu:
            menu[cat] = []
        menu[cat].append({
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "price": item.price,
            "tags": item.tags,
        })

    lines = []
    for cat, cat_items in menu.items():
        lines.append(f"\n📂 {cat.upper()}")
        for it in cat_items:
            tags = f" [{it['tags']}]" if it["tags"] else ""
            desc = f" - {it['description']}" if it["description"] else ""
            lines.append(f"  #{it['id']} {it['name']}{desc} — ₺{it['price']:.0f}{tags}")

    return "\n".join(lines)


def search_menu(db: Session, restaurant_id: int, query: str) -> str:
    query_lower = query.lower()
    items = db.query(MenuItem).filter(
        MenuItem.restaurant_id == restaurant_id,
        MenuItem.is_available == True,
    ).all()

    results = []
    for item in items:
        searchable = f"{item.name} {item.description} {item.category} {item.tags}".lower()
        if query_lower in searchable:
            results.append(item)

    if not results:
        return f"'{query}' ile eşleşen ürün bulunamadı."

    lines = [f"'{query}' araması — {len(results)} sonuç:"]
    for item in results:
        tags = f" [{item.tags}]" if item.tags else ""
        lines.append(f"  #{item.id} {item.name} — ₺{item.price:.0f}{tags}")

    return "\n".join(lines)


def add_to_cart(customer_id: str, item_id: int, quantity: int, note: str, db: Session, restaurant_id: int) -> str:
    item = db.query(MenuItem).filter(
        MenuItem.id == item_id,
        MenuItem.restaurant_id == restaurant_id,
        MenuItem.is_available == True,
    ).first()

    if not item:
        return f"#{item_id} numaralı ürün bulunamadı veya mevcut değil."

    if customer_id not in carts:
        carts[customer_id] = []

    # Aynı ürün varsa miktarını güncelle
    for cart_item in carts[customer_id]:
        if cart_item["item_id"] == item_id:
            cart_item["quantity"] += quantity
            if note:
                cart_item["note"] = note
            total = sum(ci["price"] * ci["quantity"] for ci in carts[customer_id])
            return f"✅ {item.name} x{cart_item['quantity']} güncellendi. Sepet toplamı: ₺{total:.0f}"

    carts[customer_id].append({
        "item_id": item_id,
        "name": item.name,
        "price": item.price,
        "quantity": quantity,
        "note": note or "",
    })

    total = sum(ci["price"] * ci["quantity"] for ci in carts[customer_id])
    return f"✅ {item.name} x{quantity} sepete eklendi. Sepet toplamı: ₺{total:.0f}"


def remove_from_cart(customer_id: str, item_id: int) -> str:
    if customer_id not in carts or not carts[customer_id]:
        return "Sepetiniz zaten boş."

    for i, cart_item in enumerate(carts[customer_id]):
        if cart_item["item_id"] == item_id:
            removed = carts[customer_id].pop(i)
            total = sum(ci["price"] * ci["quantity"] for ci in carts[customer_id])
            return f"🗑️ {removed['name']} sepetten çıkarıldı. Sepet toplamı: ₺{total:.0f}"

    return f"#{item_id} numaralı ürün sepetinizde yok."


def get_cart(customer_id: str) -> str:
    if customer_id not in carts or not carts[customer_id]:
        return "Sepetiniz boş."

    lines = ["🛒 Sepetiniz:"]
    total = 0
    for ci in carts[customer_id]:
        subtotal = ci["price"] * ci["quantity"]
        total += subtotal
        note_str = f" ({ci['note']})" if ci["note"] else ""
        lines.append(f"  • {ci['name']} x{ci['quantity']} — ₺{subtotal:.0f}{note_str}")

    lines.append(f"\n  💰 TOPLAM: ₺{total:.0f}")
    return "\n".join(lines)


def confirm_order(
    customer_id: str,
    delivery_address: str,
    payment_method: str,
    db: Session,
    restaurant_id: int,
) -> str:
    if customer_id not in carts or not carts[customer_id]:
        return "Sepetiniz boş, sipariş oluşturulamaz."

    cart = carts[customer_id]
    total = sum(ci["price"] * ci["quantity"] for ci in cart)

    order = Order(
        restaurant_id=restaurant_id,
        customer_phone=customer_id,
        items=json.dumps(cart, ensure_ascii=False),
        total=total,
        status="pending",
        address=delivery_address,
        payment_method=payment_method or "cash",
    )
    db.add(order)
    db.commit()
    db.refresh(order)

    # Sepeti temizle
    del carts[customer_id]

    lines = [
        f"✅ Siparişiniz alındı! (#{order.id})",
        f"",
        f"📦 Sipariş Özeti:",
    ]
    for ci in cart:
        lines.append(f"  • {ci['name']} x{ci['quantity']} — ₺{ci['price'] * ci['quantity']:.0f}")
    lines.extend([
        f"",
        f"💰 Toplam: ₺{total:.0f}",
        f"📍 Adres: {delivery_address}",
        f"💳 Ödeme: {'Nakit' if payment_method == 'cash' else 'Kapıda Kart'}",
        f"",
        f"Siparişiniz hazırlanmaya başlandığında bilgilendirileceksiniz.",
    ])

    return "\n".join(lines)


def get_order_status(customer_id: str, order_id: int, db: Session) -> str:
    order = db.query(Order).filter(
        Order.id == order_id,
        Order.customer_phone == customer_id,
    ).first()

    if not order:
        return f"#{order_id} numaralı sipariş bulunamadı."

    status_map = {
        "pending": "⏳ Onay bekliyor",
        "accepted": "👍 Kabul edildi",
        "preparing": "👨‍🍳 Hazırlanıyor",
        "ready": "✅ Hazır",
        "delivered": "🚀 Teslim edildi",
    }

    return f"Sipariş #{order.id}: {status_map.get(order.status, order.status)} | Toplam: ₺{order.total:.0f}"


def execute_tool(tool_name: str, args: dict, customer_id: str, db: Session, restaurant_id: int) -> str:
    """Tool çağrısını çalıştır ve sonucu döndür"""
    if tool_name == "get_menu":
        return get_menu(db, restaurant_id)
    elif tool_name == "search_menu":
        return search_menu(db, restaurant_id, args.get("query", ""))
    elif tool_name == "add_to_cart":
        return add_to_cart(
            customer_id, args["item_id"],
            args.get("quantity", 1), args.get("note", ""),
            db, restaurant_id,
        )
    elif tool_name == "remove_from_cart":
        return remove_from_cart(customer_id, args["item_id"])
    elif tool_name == "get_cart":
        return get_cart(customer_id)
    elif tool_name == "confirm_order":
        return confirm_order(
            customer_id, args["delivery_address"],
            args.get("payment_method", "cash"),
            db, restaurant_id,
        )
    elif tool_name == "get_order_status":
        return get_order_status(customer_id, args["order_id"], db)
    else:
        return f"Bilinmeyen tool: {tool_name}"
