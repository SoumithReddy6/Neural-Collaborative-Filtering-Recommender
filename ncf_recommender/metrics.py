from __future__ import annotations

import math


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_items: list[int], relevant_items: set[int], k: int = 10) -> float:
    relevances = [1 if item in relevant_items else 0 for item in ranked_items[:k]]
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


def hit_rate_at_k(ranked_items: list[int], relevant_items: set[int], k: int = 10) -> float:
    return 1.0 if any(item in relevant_items for item in ranked_items[:k]) else 0.0


def precision_at_k(ranked_items: list[int], relevant_items: set[int], k: int = 10) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in relevant_items)
    return hits / k


def mean_metric(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

