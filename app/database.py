from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import app.models  # noqa: F401 - register SQLAlchemy models before create_all
    Base.metadata.create_all(bind=engine)
    _ensure_conversation_columns()
    _seed_platform_defaults()


def _ensure_conversation_columns():
    """Minimal additive migration for existing databases without Alembic."""
    inspector = inspect(engine)
    if "conversations" not in inspector.get_table_names():
        return

    if engine.dialect.name == "sqlite":
        _ensure_sqlite_conversation_columns(inspector)
    elif engine.dialect.name == "postgresql":
        _ensure_postgres_conversation_columns()


def _ensure_sqlite_conversation_columns(inspector):
    existing = {column["name"] for column in inspector.get_columns("conversations")}
    additions = {
        "channel_type": "VARCHAR DEFAULT 'whatsapp'",
        "channel_account_id": "INTEGER",
        "agent_id": "INTEGER",
        "external_user_id": "VARCHAR DEFAULT ''",
        "external_msg_id": "VARCHAR DEFAULT ''",
    }

    with engine.begin() as conn:
        for name, ddl in additions.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE conversations ADD COLUMN {name} {ddl}"))


def _ensure_postgres_conversation_columns():
    statements = [
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS channel_type VARCHAR DEFAULT 'whatsapp'",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS channel_account_id INTEGER",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS agent_id INTEGER",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS external_user_id VARCHAR DEFAULT ''",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS external_msg_id VARCHAR DEFAULT ''",
        "CREATE INDEX IF NOT EXISTS ix_conversations_channel_type ON conversations (channel_type)",
        "CREATE INDEX IF NOT EXISTS ix_conversations_external_user_id ON conversations (external_user_id)",
        "CREATE INDEX IF NOT EXISTS ix_conversations_external_msg_id ON conversations (external_msg_id)",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def _seed_platform_defaults():
    import os
    from app.models import Agent, AgentKnowledgeBase, ChannelAccount, KnowledgeBase, KnowledgeDocument, Route
    from app.config import WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN

    db = SessionLocal()
    try:
        retreat = db.query(Agent).filter(Agent.slug == "retreat").first()
        if not retreat:
            retreat = Agent(
                slug="retreat",
                name="Inziva Agent",
                type="retreat",
                model="gemini-2.5-flash",
                active=True,
            )
            db.add(retreat)

        restaurant = db.query(Agent).filter(Agent.slug == "restaurant").first()
        if not restaurant:
            restaurant = Agent(
                slug="restaurant",
                name="Restoran Agent",
                type="restaurant",
                model="gemini-2.5-flash",
                active=True,
            )
            db.add(restaurant)

        db.flush()
        _ensure_default_knowledge_base(db, retreat)

        def ensure_default_routes(channel_account: ChannelAccount):
            defaults = [
                ("default", "", retreat.id, 100),
                ("prefix", "INZIVA", retreat.id, 10),
                ("prefix", "RETREAT", retreat.id, 10),
                ("prefix", "SAMMA", retreat.id, 10),
                ("prefix", "LEZZET", restaurant.id, 10),
            ]
            for match_type, match_value, agent_id, priority in defaults:
                exists = db.query(Route).filter(
                    Route.channel_account_id == channel_account.id,
                    Route.match_type == match_type,
                    Route.match_value == match_value,
                ).first()
                if not exists:
                    db.add(Route(
                        channel_account_id=channel_account.id,
                        agent_id=agent_id,
                        priority=priority,
                        match_type=match_type,
                        match_value=match_value,
                        active=True,
                    ))

        if WHATSAPP_PHONE_NUMBER_ID:
            account = db.query(ChannelAccount).filter(
                ChannelAccount.channel_type == "whatsapp",
                ChannelAccount.external_id == WHATSAPP_PHONE_NUMBER_ID,
            ).first()
            if not account:
                account = ChannelAccount(
                    channel_type="whatsapp",
                    name="Default WhatsApp",
                    external_id=WHATSAPP_PHONE_NUMBER_ID,
                    display_identifier="WhatsApp",
                    credentials_json='{"access_token_env": "WHATSAPP_ACCESS_TOKEN", "phone_number_id_env": "WHATSAPP_PHONE_NUMBER_ID"}',
                    webhook_secret=WHATSAPP_VERIFY_TOKEN,
                    active=True,
                )
                db.add(account)
                db.flush()
            ensure_default_routes(account)

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_username = os.getenv("TELEGRAM_BOT_USERNAME", "default_telegram_bot")
        if telegram_token:
            account = db.query(ChannelAccount).filter(
                ChannelAccount.channel_type == "telegram",
                ChannelAccount.external_id == telegram_username,
            ).first()
            if not account:
                account = ChannelAccount(
                    channel_type="telegram",
                    name="Default Telegram",
                    external_id=telegram_username,
                    display_identifier=telegram_username,
                    credentials_json='{"bot_token_env": "TELEGRAM_BOT_TOKEN"}',
                    active=True,
                )
                db.add(account)
                db.flush()
            ensure_default_routes(account)

        instagram_account_id = os.getenv("INSTAGRAM_ACCOUNT_ID", "default_instagram")
        if os.getenv("INSTAGRAM_ACCESS_TOKEN"):
            account = db.query(ChannelAccount).filter(
                ChannelAccount.channel_type == "instagram",
                ChannelAccount.external_id == instagram_account_id,
            ).first()
            if not account:
                account = ChannelAccount(
                    channel_type="instagram",
                    name="Default Instagram",
                    external_id=instagram_account_id,
                    display_identifier=instagram_account_id,
                    credentials_json='{"access_token_env": "INSTAGRAM_ACCESS_TOKEN"}',
                    active=True,
                )
                db.add(account)
                db.flush()
            ensure_default_routes(account)

        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"⚠️ Platform default seed hatası: {exc}")
    finally:
        db.close()


def _ensure_default_knowledge_base(db, retreat_agent):
    from pathlib import Path
    from app.models import AgentKnowledgeBase, KnowledgeBase, KnowledgeDocument

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.slug == "retreat-default").first()
    if not kb:
        kb = KnowledgeBase(
            slug="retreat-default",
            name="Retreat Knowledge",
            description="Samma Karuna inziva dokumanlari",
            active=True,
        )
        db.add(kb)
        db.flush()

    if db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id).count() == 0:
        kb_path = Path(__file__).resolve().parent.parent / "retreat_docs" / "knowledge_base.md"
        content = kb_path.read_text(encoding="utf-8") if kb_path.exists() else ""
        if content:
            db.add(KnowledgeDocument(
                knowledge_base_id=kb.id,
                filename="knowledge_base.md",
                content_type="text/markdown",
                content=content,
                source_type="seed",
                active=True,
            ))

    link = db.query(AgentKnowledgeBase).filter(
        AgentKnowledgeBase.agent_id == retreat_agent.id,
        AgentKnowledgeBase.knowledge_base_id == kb.id,
    ).first()
    if not link:
        db.add(AgentKnowledgeBase(
            agent_id=retreat_agent.id,
            knowledge_base_id=kb.id,
            priority=100,
            active=True,
        ))
