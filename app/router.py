from dataclasses import dataclass
from sqlalchemy.orm import Session
from app.models import Agent, ChannelAccount, Route


@dataclass
class RouteResolution:
    channel_account: ChannelAccount
    agent: Agent
    route: Route | None
    clean_text: str


def find_channel_account(db: Session, channel_type: str, external_id: str | None) -> ChannelAccount | None:
    if external_id:
        account = db.query(ChannelAccount).filter(
            ChannelAccount.channel_type == channel_type,
            ChannelAccount.external_id == str(external_id),
            ChannelAccount.active == True,
        ).first()
        if account:
            return account

    return db.query(ChannelAccount).filter(
        ChannelAccount.channel_type == channel_type,
        ChannelAccount.active == True,
    ).order_by(ChannelAccount.id.asc()).first()


def resolve_route(
    db: Session,
    channel_account: ChannelAccount,
    text: str,
    metadata: dict | None = None,
) -> RouteResolution | None:
    metadata = metadata or {}
    routes = db.query(Route).join(Agent).filter(
        Route.channel_account_id == channel_account.id,
        Route.active == True,
        Agent.active == True,
    ).order_by(Route.priority.asc(), Route.id.asc()).all()

    default_route = None
    for route in routes:
        if route.match_type == "default":
            default_route = default_route or route
            continue

        if _matches(route, text, metadata):
            return RouteResolution(
                channel_account=channel_account,
                agent=route.agent,
                route=route,
                clean_text=_clean_text(route, text),
            )

    if default_route:
        return RouteResolution(
            channel_account=channel_account,
            agent=default_route.agent,
            route=default_route,
            clean_text=text,
        )

    return None


def _matches(route: Route, text: str, metadata: dict) -> bool:
    value = (route.match_value or "").strip()
    text_clean = (text or "").strip()
    text_upper = text_clean.upper()
    value_upper = value.upper()

    if route.match_type == "prefix":
        first_word = text_upper.split()[0] if text_upper else ""
        return bool(value_upper) and first_word.startswith(value_upper)
    if route.match_type == "exact":
        return text_upper == value_upper
    if route.match_type == "keyword":
        return bool(value_upper) and value_upper in text_upper
    if route.match_type == "ad_source":
        return bool(value) and str(metadata.get("ad_source", "")).upper() == value_upper

    return False


def _clean_text(route: Route, text: str) -> str:
    if route.match_type != "prefix":
        return text

    stripped = text.strip()
    first_word = stripped.split()[0] if stripped else ""
    value = (route.match_value or "").strip().upper()
    if value and first_word.upper().startswith(value):
        clean = stripped[len(first_word):].strip()
        return clean or "Merhaba"
    return text
