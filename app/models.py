from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey, DateTime, Date
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class ChannelAccount(Base):
    __tablename__ = "channel_accounts"

    id = Column(Integer, primary_key=True)
    channel_type = Column(String, nullable=False)  # whatsapp, telegram, instagram
    name = Column(String, nullable=False)
    external_id = Column(String, nullable=False, index=True)  # phone_number_id, bot username/id, page/account id
    display_identifier = Column(String, default="")
    credentials_json = Column(Text, default="{}")  # env/secret references, not raw secrets
    webhook_secret = Column(String, default="")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    routes = relationship("Route", back_populates="channel_account")


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True)
    slug = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # generic_prompt, retreat, restaurant, custom_code
    model = Column(String, default="gemini-2.5-flash")
    fallback_model = Column(String, default="")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, nullable=True)
    timeout_seconds = Column(Integer, default=60)
    daily_budget_limit = Column(Float, nullable=True)
    monthly_budget_limit = Column(Float, nullable=True)
    failover_enabled = Column(Boolean, default=True)
    system_prompt = Column(Text, default="")
    knowledge_base = Column(Text, default="")
    rag_top_k = Column(Integer, default=20)
    rag_final_chunks = Column(Integer, default=6)
    rag_min_score = Column(Float, nullable=True)
    rag_max_context_chars = Column(Integer, default=12000)
    rag_hybrid_search = Column(Boolean, default=True)
    rag_rerank_enabled = Column(Boolean, default=False)
    rag_query_rewrite_enabled = Column(Boolean, default=False)
    rag_embedding_model = Column(String, default="")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    routes = relationship("Route", back_populates="agent")
    knowledge_links = relationship("AgentKnowledgeBase", back_populates="agent")


class LlmProvider(Base):
    __tablename__ = "llm_providers"

    id = Column(Integer, primary_key=True)
    slug = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    base_url = Column(String, default="")
    api_key_env = Column(String, default="")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    models = relationship("LlmModel", back_populates="provider")


class LlmModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True)
    provider_id = Column(Integer, ForeignKey("llm_providers.id"), nullable=False, index=True)
    slug = Column(String, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    model_ref = Column(String, nullable=False, unique=True, index=True)
    supports_tools = Column(Boolean, default=False)
    supports_vision = Column(Boolean, default=False)
    context_window = Column(Integer, nullable=True)
    input_price = Column(Float, nullable=True)
    output_price = Column(Float, nullable=True)
    pricing_source = Column(String, default="")
    pricing_checked_at = Column(DateTime, nullable=True)
    pricing_sync_error = Column(Text, default="")
    rate_limit_rpm = Column(Integer, nullable=True)
    rate_limit_tpm = Column(Integer, nullable=True)
    active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    provider = relationship("LlmProvider", back_populates="models")


class LlmUsageLog(Base):
    __tablename__ = "llm_usage_logs"

    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True, index=True)
    source = Column(String, default="", index=True)
    operation = Column(String, default="", index=True)
    provider = Column(String, default="", index=True)
    model_ref = Column(String, default="", index=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    estimated_cost = Column(Float, nullable=True)
    actual_cost = Column(Float, nullable=True)
    generation_id = Column(String, default="")
    actual_model_ref = Column(String, default="")
    actual_provider = Column(String, default="")
    router = Column(String, default="")
    cost_details_json = Column(Text, default="")
    latency_ms = Column(Integer, nullable=True)
    success = Column(Boolean, default=True, index=True)
    error_code = Column(String, default="")
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    agent = relationship("Agent")


class Route(Base):
    __tablename__ = "routes"

    id = Column(Integer, primary_key=True)
    channel_account_id = Column(Integer, ForeignKey("channel_accounts.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    priority = Column(Integer, default=100)
    match_type = Column(String, nullable=False, default="default")  # default, prefix, keyword, ad_source, exact
    match_value = Column(String, default="")
    active = Column(Boolean, default=True)

    channel_account = relationship("ChannelAccount", back_populates="routes")
    agent = relationship("Agent", back_populates="routes")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True)
    slug = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    documents = relationship("KnowledgeDocument", back_populates="knowledge_base")
    agent_links = relationship("AgentKnowledgeBase", back_populates="knowledge_base")


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, default="text/plain")
    content = Column(Text, nullable=False)
    source_type = Column(String, default="upload")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("KnowledgeChunk", back_populates="document")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("knowledge_documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    title_path = Column(Text, default="")
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False, index=True)
    token_count = Column(Integer, default=0)
    metadata_json = Column(Text, default="{}")
    active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    knowledge_base = relationship("KnowledgeBase")
    document = relationship("KnowledgeDocument", back_populates="chunks")
    embeddings = relationship("KnowledgeEmbedding", back_populates="chunk")


class KnowledgeEmbedding(Base):
    __tablename__ = "knowledge_embeddings"

    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer, ForeignKey("knowledge_chunks.id"), nullable=False, index=True)
    embedding_model = Column(String, nullable=False, index=True)
    embedding_dim = Column(Integer, nullable=False)
    vector_json = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False, index=True)
    status = Column(String, default="ready", index=True)
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    chunk = relationship("KnowledgeChunk", back_populates="embeddings")


class RagQueryLog(Base):
    __tablename__ = "rag_query_logs"

    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True, index=True)
    external_user_id = Column(String, default="", index=True)
    query = Column(Text, nullable=False)
    rewritten_query = Column(Text, default="")
    retrieved_chunk_ids_json = Column(Text, default="[]")
    scores_json = Column(Text, default="[]")
    selected_context_chars = Column(Integer, default=0)
    retrieval_latency_ms = Column(Integer, nullable=True)
    rerank_latency_ms = Column(Integer, nullable=True)
    answer_latency_ms = Column(Integer, nullable=True)
    model_ref = Column(String, default="")
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    agent = relationship("Agent")


class AgentKnowledgeBase(Base):
    __tablename__ = "agent_knowledge_bases"

    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    priority = Column(Integer, default=100)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    agent = relationship("Agent", back_populates="knowledge_links")
    knowledge_base = relationship("KnowledgeBase", back_populates="agent_links")


class ImageLocalizationJob(Base):
    __tablename__ = "image_localization_jobs"

    id = Column(Integer, primary_key=True)
    target_language = Column(String, default="tr")
    status = Column(String, default="created", index=True)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    assets = relationship("ImageLocalizationAsset", back_populates="job")


class ImageLocalizationAsset(Base):
    __tablename__ = "image_localization_assets"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("image_localization_jobs.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, default="image/png")
    original_path = Column(Text, nullable=False)
    cropped_path = Column(Text, default="")
    output_path = Column(Text, default="")
    crop_json = Column(Text, default="{}")
    ocr_json = Column(Text, default="[]")
    translations_json = Column(Text, default="[]")
    approved_texts_json = Column(Text, default="[]")
    status = Column(String, default="uploaded", index=True)
    error_message = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    job = relationship("ImageLocalizationJob", back_populates="assets")


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
    channel_type = Column(String, default="whatsapp", index=True)
    channel_account_id = Column(Integer, ForeignKey("channel_accounts.id"), nullable=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    external_user_id = Column(String, default="", index=True)
    customer_phone = Column(String, nullable=False)
    customer_name = Column(String, default="")
    agent_type = Column(String, nullable=False)  # retreat, restaurant
    role = Column(String, nullable=False)  # user, agent
    message = Column(Text, nullable=False)
    external_msg_id = Column(String, default="", index=True)
    msg_id = Column(String, default="")  # WhatsApp message ID (duplicate check)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    channel_account = relationship("ChannelAccount")
    agent = relationship("Agent")


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
