import os
import tempfile
import unittest

db_file = tempfile.NamedTemporaryFile(prefix="rag-test-", suffix=".db", delete=False)
db_file.close()
os.environ["DATABASE_URL"] = f"sqlite:///{db_file.name}"
os.environ["RAG_EMBEDDING_MODEL"] = "local:hash-v1"

from app.agent_registry import build_agent_prompt_knowledge
from app.database import SessionLocal, init_db
from app.models import Agent, KnowledgeBase, KnowledgeChunk, KnowledgeDocument, KnowledgeEmbedding, LlmUsageLog, RagQueryLog
from app.rag.indexing import index_knowledge_base
from app.rag.retrieval import retrieve_agent_context
import app.rag.indexing as indexing_module
import app.rag.retrieval as retrieval_module


class RagMvpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_demo_agent_retrieves_indexed_knowledge(self):
        agent = self.db.query(Agent).filter(Agent.slug == "rag-demo").first()
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.slug == "rag-demo-knowledge").first()

        self.assertIsNotNone(agent)
        self.assertIsNotNone(kb)

        index_result = index_knowledge_base(kb.id, self.db)
        self.db.commit()

        self.assertTrue(index_result["ok"])
        self.assertGreaterEqual(index_result["chunk_count"], 3)
        self.assertGreater(
            self.db.query(KnowledgeChunk).filter(KnowledgeChunk.knowledge_base_id == kb.id, KnowledgeChunk.active == True).count(),
            0,
        )
        self.assertGreater(
            self.db.query(KnowledgeEmbedding).filter(KnowledgeEmbedding.status == "ready").count(),
            0,
        )

        result = retrieve_agent_context(
            agent,
            "Nova Bakim Paketi fiyati nedir?",
            self.db,
            external_user_id="rag-e2e-test",
        )
        self.db.commit()

        self.assertGreater(len(result.chunks), 0)
        self.assertIn("4.750 TL", result.context)
        self.assertGreater(
            self.db.query(RagQueryLog).filter(RagQueryLog.external_user_id == "rag-e2e-test").count(),
            0,
        )

    def test_agent_prompt_uses_rag_context(self):
        agent = self.db.query(Agent).filter(Agent.slug == "rag-demo").first()
        knowledge = build_agent_prompt_knowledge(
            agent,
            "rag-e2e-prompt",
            "Nova Bakim Paketi kapsaminda neler var?",
            self.db,
        )

        self.assertIn("RAG ILE SECILEN ILGILI BILGI PARCALARI", knowledge)
        self.assertIn("haftalik cevap kalitesi incelemesi", knowledge.lower())

    def test_embedding_calls_are_logged_for_non_local_models(self):
        agent = self.db.query(Agent).filter(Agent.slug == "rag-demo").first()
        kb = self.db.query(KnowledgeBase).filter(KnowledgeBase.slug == "rag-demo-knowledge").first()
        doc = KnowledgeDocument(
            knowledge_base_id=kb.id,
            filename="usage-log-test.md",
            content_type="text/markdown",
            content="# Log Test\n\nEmbedding usage log test 19.250 TL fiyat bilgisidir.",
            source_type="test",
            active=True,
        )
        self.db.add(doc)
        self.db.flush()

        original_embed_texts = indexing_module.embed_texts
        original_embed_text = retrieval_module.embed_text
        try:
            indexing_module.embed_texts = lambda texts, model_ref=None: [[1.0, 0.0, 0.0] for _ in texts]
            retrieval_module.embed_text = lambda text, model_ref=None: [1.0, 0.0, 0.0]
            indexing_module.index_document(doc, self.db, model_ref="openai:test-embedding", agent_id=agent.id)
            agent.rag_embedding_model = "openai:test-embedding"
            result = retrieve_agent_context(agent, "Log test fiyati nedir?", self.db, external_user_id="usage-log-test")
            self.db.commit()
        finally:
            indexing_module.embed_texts = original_embed_texts
            retrieval_module.embed_text = original_embed_text

        self.assertIn("19.250 TL", result.context)
        operations = {
            row.operation
            for row in self.db.query(LlmUsageLog).filter(
                LlmUsageLog.source == "rag",
                LlmUsageLog.model_ref == "openai:test-embedding",
            ).all()
        }
        self.assertIn("embedding:index", operations)
        self.assertIn("embedding:query", operations)


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        try:
            os.unlink(db_file.name)
        except OSError:
            pass
