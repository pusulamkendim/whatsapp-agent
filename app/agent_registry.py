from sqlalchemy.orm import Session
from app.models import Agent, AgentKnowledgeBase, KnowledgeDocument
from app.agent import chat as restaurant_chat
from app.retreat_agent import chat as retreat_chat
from app.llm import is_gemini_model, parse_model_ref, run_openai_simple_chat, GEMINI_CLIENT
from google.genai import types


RESTAURANT_ID = 1
RESTAURANT_NAME = "Lezzet Durağı"


def run_agent(agent: Agent, customer_id: str, message: str, db: Session) -> str:
    if agent.type == "restaurant" or agent.slug == "restaurant":
        return restaurant_chat(customer_id, message, RESTAURANT_ID, RESTAURANT_NAME, db, model=agent.model)

    if agent.type == "retreat" or agent.slug == "retreat":
        knowledge = build_agent_knowledge(agent, db)
        return retreat_chat(customer_id, message, knowledge_base=knowledge or None, model=agent.model)

    if agent.type == "generic_prompt":
        knowledge = build_agent_knowledge(agent, db)
        return _generic_prompt_response(agent, customer_id, message, knowledge)

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


generic_conversations: dict[str, list[dict]] = {}
generic_gemini_conversations: dict[str, list] = {}


def _generic_prompt_response(agent: Agent, customer_id: str, message: str, knowledge: str) -> str:
    prompt = agent.system_prompt or f"Sen {agent.name} isimli yardımcı bir asistansın."
    if knowledge:
        prompt = f"{prompt}\n\nBILGI BANKASI:\n{knowledge}"

    conversation_key = f"{agent.id}:{customer_id}"
    if is_gemini_model(agent.model):
        if conversation_key not in generic_gemini_conversations:
            generic_gemini_conversations[conversation_key] = []
        history = generic_gemini_conversations[conversation_key]
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        response = GEMINI_CLIENT.models.generate_content(
            model=parse_model_ref(agent.model)[1],
            contents=history,
            config=types.GenerateContentConfig(system_instruction=prompt, temperature=0.7),
        )
        text = response.candidates[0].content.parts[0].text
        history.append(response.candidates[0].content)
        if len(history) > 30:
            generic_gemini_conversations[conversation_key] = history[-30:]
        return text

    if conversation_key not in generic_conversations:
        generic_conversations[conversation_key] = [{"role": "system", "content": prompt}]
    history = generic_conversations[conversation_key]
    history.append({"role": "user", "content": message})
    text = run_openai_simple_chat(agent.model, history)
    if len(history) > 32:
        generic_conversations[conversation_key] = [history[0], *history[-31:]]
    return text
