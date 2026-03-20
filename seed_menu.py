"""Test restoran ve menü verisi yükle"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db, SessionLocal
from app.models import Restaurant, MenuItem

def seed():
    init_db()
    db = SessionLocal()

    # Mevcut veriyi temizle
    db.query(MenuItem).delete()
    db.query(Restaurant).delete()
    db.commit()

    # Test restoran
    restaurant = Restaurant(
        id=1,
        name="Lezzet Durağı",
        phone="0242 259 00 00",
        address="Hurma Mah. Boğaçay Cad. No:42, Konyaaltı/Antalya",
        whatsapp_number="905321234567",
    )
    db.add(restaurant)
    db.commit()

    # Menü
    items = [
        # Kebaplar
        MenuItem(restaurant_id=1, name="Adana Kebap", description="Acılı el yapımı kebap, lavash ve közlenmiş domates ile", price=220, category="Kebaplar", tags="acılı,et"),
        MenuItem(restaurant_id=1, name="Urfa Kebap", description="Acısız el yapımı kebap", price=220, category="Kebaplar", tags="et"),
        MenuItem(restaurant_id=1, name="İskender", description="Tereyağlı domates soslu, yoğurtlu", price=280, category="Kebaplar", tags="et"),
        MenuItem(restaurant_id=1, name="Tavuk Şiş", description="Marine edilmiş tavuk göğsü", price=180, category="Kebaplar", tags="tavuk"),
        MenuItem(restaurant_id=1, name="Beyti Sarma", description="Lavash ile sarılmış kebap, yoğurt ve sos", price=260, category="Kebaplar", tags="et"),
        MenuItem(restaurant_id=1, name="Kuzu Pirzola", description="Izgara kuzu pirzola (4 adet)", price=380, category="Kebaplar", tags="et"),

        # Pideler
        MenuItem(restaurant_id=1, name="Kıymalı Pide", description="Taze kıyma ile", price=160, category="Pideler", tags="et"),
        MenuItem(restaurant_id=1, name="Kaşarlı Pide", description="Bol kaşar peyniri", price=140, category="Pideler", tags="vejetaryen"),
        MenuItem(restaurant_id=1, name="Karışık Pide", description="Kıyma, kaşar, sucuk", price=180, category="Pideler", tags="et"),
        MenuItem(restaurant_id=1, name="Kuşbaşılı Pide", description="Dana kuşbaşı ile", price=200, category="Pideler", tags="et"),
        MenuItem(restaurant_id=1, name="Mantar Pide", description="Mantar ve kaşar peyniri", price=150, category="Pideler", tags="vejetaryen"),

        # Lahmacun
        MenuItem(restaurant_id=1, name="Lahmacun", description="İnce hamur, kıymalı (adet)", price=60, category="Lahmacun", tags="et,acılı"),
        MenuItem(restaurant_id=1, name="Lahmacun Dürüm", description="Lahmacun dürüm yapılmış", price=70, category="Lahmacun", tags="et,acılı"),

        # Salatalar
        MenuItem(restaurant_id=1, name="Çoban Salata", description="Domates, salatalık, biber, soğan", price=60, category="Salatalar", tags="vejetaryen,hafif"),
        MenuItem(restaurant_id=1, name="Mevsim Salata", description="Mevsim yeşillikleri", price=70, category="Salatalar", tags="vejetaryen,hafif"),
        MenuItem(restaurant_id=1, name="Acılı Ezme", description="Bol acılı ezme porsiyon", price=50, category="Salatalar", tags="vejetaryen,acılı"),

        # Mezeler
        MenuItem(restaurant_id=1, name="Humus", description="Nohut ezmesi, zeytinyağı", price=60, category="Mezeler", tags="vejetaryen"),
        MenuItem(restaurant_id=1, name="Babaganuş", description="Közlenmiş patlıcan ezmesi", price=60, category="Mezeler", tags="vejetaryen"),
        MenuItem(restaurant_id=1, name="Sigara Böreği", description="Peynirli (4 adet)", price=70, category="Mezeler", tags="vejetaryen"),
        MenuItem(restaurant_id=1, name="Patlıcan Kızartma", description="Yoğurtlu patlıcan", price=65, category="Mezeler", tags="vejetaryen"),

        # İçecekler
        MenuItem(restaurant_id=1, name="Ayran", description="Taze ayran", price=25, category="İçecekler", tags=""),
        MenuItem(restaurant_id=1, name="Kola", description="330ml", price=35, category="İçecekler", tags=""),
        MenuItem(restaurant_id=1, name="Salgam", description="Acılı veya acısız", price=25, category="İçecekler", tags=""),
        MenuItem(restaurant_id=1, name="Su", description="0.5L", price=10, category="İçecekler", tags=""),
        MenuItem(restaurant_id=1, name="Çay", description="Bardak çay", price=15, category="İçecekler", tags=""),

        # Tatlılar
        MenuItem(restaurant_id=1, name="Künefe", description="Sıcak servis, antep fıstıklı", price=120, category="Tatlılar", tags=""),
        MenuItem(restaurant_id=1, name="Katmer", description="Antep fıstıklı katmer", price=110, category="Tatlılar", tags=""),
        MenuItem(restaurant_id=1, name="Sütlaç", description="Fırında sütlaç", price=60, category="Tatlılar", tags=""),
    ]

    db.add_all(items)
    db.commit()
    db.close()

    print(f"✅ Restoran ve {len(items)} menü öğesi yüklendi.")


if __name__ == "__main__":
    seed()
