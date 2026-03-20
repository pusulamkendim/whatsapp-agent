from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey, DateTime, Date
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    phone = Column(String)
    address = Column(String)
    whatsapp_number = Column(String)

    menu_items = relationship("MenuItem", back_populates="restaurant")
    orders = relationship("Order", back_populates="restaurant")


class MenuItem(Base):
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    price = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    is_available = Column(Boolean, default=True)
    tags = Column(String, default="")  # comma-separated: acılı,vejetaryen

    restaurant = relationship("Restaurant", back_populates="menu_items")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False)
    customer_phone = Column(String, nullable=False)
    items = Column(Text, nullable=False)  # JSON
    total = Column(Float, nullable=False)
    status = Column(String, default="pending")  # pending/accepted/preparing/ready/delivered
    address = Column(String, nullable=False)
    payment_method = Column(String, default="cash")
    note = Column(String, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    restaurant = relationship("Restaurant", back_populates="orders")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    customer_phone = Column(String, nullable=False)
    customer_name = Column(String, default="")
    agent_type = Column(String, nullable=False)  # retreat, restaurant
    role = Column(String, nullable=False)  # user, agent
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Handoff(Base):
    __tablename__ = "handoffs"

    id = Column(Integer, primary_key=True)
    customer_phone = Column(String, nullable=False)
    customer_name = Column(String, default="")
    conversation_summary = Column(Text, default="")
    interest_level = Column(String, default="")
    status = Column(String, default="pending")  # pending, contacted, converted, lost
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DailyStat(Base):
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)
    total_messages = Column(Integer, default=0)
    user_messages = Column(Integer, default=0)
    agent_messages = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    new_users = Column(Integer, default=0)  # ilk kez yazan
    handoffs = Column(Integer, default=0)
    agent_type_breakdown = Column(Text, default="{}")  # JSON: {"retreat": 10, "restaurant": 5}
