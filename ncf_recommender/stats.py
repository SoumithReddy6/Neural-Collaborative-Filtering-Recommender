from __future__ import annotations

from collections import defaultdict

from ncf_recommender.data import Interaction


def click_through_rate(interactions: list[Interaction]) -> float:
    if not interactions:
        return 0.0
    return sum(row.clicked for row in interactions) / len(interactions)


def average_session_depth(interactions: list[Interaction]) -> float:
    sessions: dict[str, int] = defaultdict(int)
    for row in interactions:
        sessions[row.session_id] += 1
    return sum(sessions.values()) / len(sessions) if sessions else 0.0


def interaction_summary(interactions: list[Interaction]) -> dict[str, float]:
    return {
        "num_interactions": float(len(interactions)),
        "num_users": float(len({row.user_id for row in interactions})),
        "num_items": float(len({row.item_id for row in interactions})),
        "ctr": click_through_rate(interactions),
        "avg_session_depth": average_session_depth(interactions),
    }

