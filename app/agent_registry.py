from sqlalchemy.orm import Session
from app.models import Agent
from app.agent import chat as restaurant_chat
from app.retreat_agent import chat as retreat_chat


RESTAURANT_ID = 1
RESTAURANT_NAME = "Lezzet Durağı"


def run_agent(agent: Agent, customer_id: str, message: str, db: Session) -> str:
    if agent.type == "restaurant" or agent.slug == "restaurant":
        return restaurant_chat(customer_id, message, RESTAURANT_ID, RESTAURANT_NAME, db)

    if agent.type == "retreat" or agent.slug == "retreat":
        return retreat_chat(customer_id, message)

    if agent.type == "generic_prompt":
        return _generic_prompt_response(agent)

    return "Bu agent tipi henüz desteklenmiyor."


def _generic_prompt_response(agent: Agent) -> str:
    # Placeholder until generic prompt agents get their own Gemini wrapper.
    return f"{agent.name} için generic prompt çalıştırıcı henüz yapılandırılmadı."
