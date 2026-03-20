"""Terminal üzerinden agent'ı test et (WhatsApp olmadan)"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db, SessionLocal
from app.agent import chat

RESTAURANT_ID = 1
RESTAURANT_NAME = "Lezzet Durağı"
CUSTOMER_ID = "test_user"


def main():
    init_db()

    print("=" * 50)
    print(f"  {RESTAURANT_NAME} - WhatsApp Sipariş Agent")
    print("  (Çıkmak için 'q' yazın)")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit", "çık"):
            print("Görüşürüz! 👋")
            break

        db = SessionLocal()
        try:
            response = chat(CUSTOMER_ID, user_input, RESTAURANT_ID, RESTAURANT_NAME, db)
            print(f"\n🤖 Agent: {response}\n")
        except Exception as e:
            print(f"\n❌ Hata: {e}\n")
        finally:
            db.close()


if __name__ == "__main__":
    main()
