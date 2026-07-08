from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False, "timeout": 30}

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
    _run_alembic_migrations()
    _ensure_conversation_columns()
    _ensure_rag_columns()
    _seed_platform_defaults()
    _seed_llm_catalog()


def _run_alembic_migrations():
    try:
        from alembic import command
        from alembic.config import Config

        config = Config("alembic.ini")
        command.upgrade(config, "head")
    except Exception as exc:
        print(f"⚠️ Alembic migration hatası: {exc}")


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


def _ensure_rag_columns():
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "agents" in tables:
        _ensure_table_columns("agents", {
            "rag_top_k": "INTEGER DEFAULT 20",
            "rag_final_chunks": "INTEGER DEFAULT 6",
            "rag_min_score": "FLOAT",
            "rag_max_context_chars": "INTEGER DEFAULT 12000",
            "rag_hybrid_search": "BOOLEAN DEFAULT TRUE",
            "rag_rerank_enabled": "BOOLEAN DEFAULT FALSE",
            "rag_query_rewrite_enabled": "BOOLEAN DEFAULT FALSE",
            "rag_embedding_model": "VARCHAR DEFAULT ''",
        })
    if "rag_query_logs" in tables:
        _ensure_table_columns("rag_query_logs", {
            "answer_latency_ms": "INTEGER",
            "model_ref": "VARCHAR DEFAULT ''",
            "prompt_tokens": "INTEGER",
            "completion_tokens": "INTEGER",
            "total_tokens": "INTEGER",
        })


def _ensure_table_columns(table_name: str, additions: dict[str, str]):
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    with engine.begin() as conn:
        for name, ddl in additions.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {name} {ddl}"))


def _seed_platform_defaults():
    import os
    from app.models import Agent, AgentKnowledgeBase, ChannelAccount, KnowledgeBase, KnowledgeDocument, Route
    from app.config import WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN

    db = SessionLocal()
    try:
        with db.no_autoflush:
            retreat = db.query(Agent).filter(Agent.slug == "retreat").first()
            if not retreat:
                retreat = Agent(
                    slug="retreat",
                    name="Inziva Agent",
                    type="retreat",
                    model="gemini:gemini-2.5-flash",
                    active=True,
                )
                db.add(retreat)

            restaurant = db.query(Agent).filter(Agent.slug == "restaurant").first()
            if not restaurant:
                restaurant = Agent(
                    slug="restaurant",
                    name="Restoran Agent",
                    type="restaurant",
                    model="gemini:gemini-2.5-flash",
                    active=True,
                )
                db.add(restaurant)

            rag_demo = db.query(Agent).filter(Agent.slug == "rag-demo").first()
            if not rag_demo:
                rag_demo = Agent(
                    slug="rag-demo",
                    name="RAG Demo Agent",
                    type="generic_prompt",
                    model="gemini:gemini-2.5-flash",
                    system_prompt=(
                        "Sen bilgi bankasina dayali cevap veren kisa ve net bir asistansin. "
                        "Sadece sana verilen RAG kaynaklarindaki bilgileri kullan. "
                        "Kaynaklarda yeterli bilgi yoksa 'Bu bilgi bilgi bankasinda yok' de. "
                        "Cevaplari Turkce ver."
                    ),
                    active=True,
                )
                db.add(rag_demo)

            db.flush()
            _ensure_default_knowledge_base(db, retreat)
            _ensure_rag_demo_knowledge_base(db, rag_demo)

            def ensure_default_routes(channel_account: ChannelAccount):
                defaults = [
                    ("default", "", retreat.id, 100),
                    ("prefix", "INZIVA", retreat.id, 10),
                    ("prefix", "RETREAT", retreat.id, 10),
                    ("prefix", "SAMMA", retreat.id, 10),
                    ("prefix", "LEZZET", restaurant.id, 10),
                    ("prefix", "RAG", rag_demo.id, 10),
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


def _seed_llm_catalog():
    from datetime import datetime, timezone
    from app.llm import MODEL_OPTIONS
    from app.models import LlmModel, LlmProvider

    provider_defaults = {
        "gemini": {
            "name": "Google Gemini",
            "base_url": "",
            "api_key_env": "GEMINI_API_KEY",
        },
        "deepseek": {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "openrouter": {
            "name": "OpenRouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "ollama": {
            "name": "Ollama",
            "base_url": "http://localhost:11434/v1",
            "api_key_env": "OLLAMA_BASE_URL",
        },
        "openai": {
            "name": "OpenAI Compatible",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
        "cerebras": {
            "name": "Cerebras",
            "base_url": "https://api.cerebras.ai/v1",
            "api_key_env": "CEREBRAS_API_KEY",
        },
    }

    db = SessionLocal()
    try:
        providers: dict[str, LlmProvider] = {}
        with db.no_autoflush:
            for slug, defaults in provider_defaults.items():
                provider = db.query(LlmProvider).filter(LlmProvider.slug == slug).first()
                if not provider:
                    provider = LlmProvider(slug=slug, **defaults, active=True)
                    db.add(provider)
                providers[slug] = provider

            db.flush()

            for option in MODEL_OPTIONS:
                provider_slug = option["provider"]
                provider = providers.get(provider_slug)
                if not provider:
                    continue
                model = db.query(LlmModel).filter(LlmModel.model_ref == option["model"]).first()
                if not model:
                    model = LlmModel(
                        provider_id=provider.id,
                        slug=option["model"].split(":", 1)[1],
                        display_name=option["label"],
                        model_ref=option["model"],
                        supports_tools=provider_slug in {"gemini", "openrouter", "deepseek", "openai", "ollama", "cerebras"},
                        supports_vision=provider_slug in {"gemini", "openrouter", "openai"} or option["model"] == "cerebras:gemma-4-31b",
                        is_default=option["model"] == "gemini:gemini-2.5-flash",
                        notes=option.get("notes", ""),
                        active=True,
                    )
                    db.add(model)
                else:
                    model.display_name = model.display_name or option["label"]
                    model.notes = model.notes or option.get("notes", "")
                    model.updated_at = datetime.now(timezone.utc)

        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"⚠️ LLM catalog seed hatası: {exc}")
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


def _ensure_rag_demo_knowledge_base(db, rag_agent):
    from app.models import AgentKnowledgeBase, KnowledgeBase, KnowledgeDocument

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.slug == "rag-demo-knowledge").first()
    if not kb:
        kb = KnowledgeBase(
            slug="rag-demo-knowledge",
            name="RAG Demo Knowledge",
            description="RAG MVP test dokumani",
            active=True,
        )
        db.add(kb)
        db.flush()

    if db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == kb.id).count() == 0:
        db.add(KnowledgeDocument(
            knowledge_base_id=kb.id,
            filename="rag-demo.md",
            content_type="text/markdown",
            content=(
                "# Nova Bakim Paketi\n\n"
                "Nova Bakim Paketi, sadece RAG demo agent testleri icin tanimlanan kurumsal destek paketidir.\n\n"
                "## Fiyat\n\n"
                "Nova Bakim Paketi aylik 4.750 TL olarak tanimlanmistir.\n\n"
                "## Kapsam\n\n"
                "Paket; bilgi bankasi guncelleme kontrolu, haftalik cevap kalitesi incelemesi ve "
                "en fazla 3 yeni dokuman indeksleme destegi icerir.\n\n"
                "## Dahil Olmayanlar\n\n"
                "Ozel yazilim gelistirme, reklam yonetimi ve canli operator hizmeti bu pakete dahil degildir.\n"
            ),
            source_type="seed",
            active=True,
        ))

    link = db.query(AgentKnowledgeBase).filter(
        AgentKnowledgeBase.agent_id == rag_agent.id,
        AgentKnowledgeBase.knowledge_base_id == kb.id,
    ).first()
    if not link:
        db.add(AgentKnowledgeBase(
            agent_id=rag_agent.id,
            knowledge_base_id=kb.id,
            priority=100,
            active=True,
        ))
