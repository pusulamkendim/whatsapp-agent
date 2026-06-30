from sqlalchemy.orm import Session
from app.models import Agent, AgentKnowledgeBase, KnowledgeDocument
from app.agent import chat as restaurant_chat
from app.retreat_agent import chat as retreat_chat


RESTAURANT_ID = 1
RESTAURANT_NAME = "Lezzet Durağı"


def run_agent(agent: Agent, customer_id: str, message: str, db: Session) -> str:
    if agent.type == "restaurant" or agent.slug == "restaurant":
        return restaurant_chat(customer_id, message, RESTAURANT_ID, RESTAURANT_NAME, db)

    if agent.type == "retreat" or agent.slug == "retreat":
        knowledge = build_agent_knowledge(agent, db)
        return retreat_chat(customer_id, message, knowledge_base=knowledge or None)

    if agent.type == "generic_prompt":
        return _generic_prompt_response(agent)

    return "Bu agent tipi henüz desteklenmiyor."


def build_agent_knowledge(agent: Agent, db: Session) -> str:
    links = db.query(AgentKnowledgeBase).filter(
        AgentKnowledgeBase.agent_id == agent.id,
        AgentKnowledgeBase.active == True,
    ).order_by(AgentKnowledgeBase.priority.asc(), AgentKnowledgeBase.id.asc()).all()

    sections = []
    for link in links:
        kb = link.knowledge_base
        if not kb or not kb.active:
            continue
        docs = db.query(KnowledgeDocument).filter(
            KnowledgeDocument.knowledge_base_id == kb.id,
            KnowledgeDocument.active == True,
        ).order_by(KnowledgeDocument.id.asc()).all()
        if not docs:
            continue
        doc_parts = [f"# {kb.name}"]
        if kb.description:
            doc_parts.append(kb.description)
        for doc in docs:
            doc_parts.append(f"## {doc.filename}\n{doc.content}")
        sections.append("\n\n".join(doc_parts))

    if agent.knowledge_base:
        sections.append(f"# {agent.name} inline knowledge\n{agent.knowledge_base}")

    return "\n\n---\n\n".join(sections)


def _generic_prompt_response(agent: Agent) -> str:
    # Placeholder until generic prompt agents get their own Gemini wrapper.
    return f"{agent.name} için generic prompt çalıştırıcı henüz yapılandırılmadı."
