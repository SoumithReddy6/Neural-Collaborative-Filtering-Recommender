from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ncf_recommender.losses import bpr_loss
from ncf_recommender.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, precision_at_k
from ncf_recommender.models import MatrixFactorization, NeuralCollaborativeFiltering, score_all_items


@dataclass
class TrainingResult:
    model: nn.Module
    losses: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_type: str, num_users: int, num_items: int, embedding_dim: int = 32) -> nn.Module:
    if model_type == "mf":
        return MatrixFactorization(num_users, num_items, embedding_dim)
    if model_type == "ncf":
        return NeuralCollaborativeFiltering(num_users, num_items, embedding_dim)
    raise ValueError(f"Unsupported model_type: {model_type}")


def train_bpr(
    model: nn.Module,
    triples: list[tuple[int, int, int]],
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> TrainingResult:
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    triples = list(triples)
    for _ in range(epochs):
        random.shuffle(triples)
        batch_losses: list[float] = []
        for start in range(0, len(triples), batch_size):
            batch = triples[start : start + batch_size]
            users = torch.tensor([row[0] for row in batch], dtype=torch.long)
            positives = torch.tensor([row[1] for row in batch], dtype=torch.long)
            negatives = torch.tensor([row[2] for row in batch], dtype=torch.long)
            optimizer.zero_grad()
            loss = bpr_loss(model(users, positives), model(users, negatives))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        losses.append(mean_metric(batch_losses))
    return TrainingResult(model=model, losses=losses)


def evaluate_ranking_sampled(
    model: nn.Module,
    test_pairs: list[tuple[int, int]],
    train_pairs: list[tuple[int, int]],
    num_items: int,
    k: int = 10,
    num_negatives: int = 99,
    seed: int = 42,
) -> dict[str, float]:
    """Standard NCF (He et al. 2017) sampled evaluation protocol.

    For each held-out (user, item), rank the positive against `num_negatives`
    randomly sampled items the user has not interacted with, then compute HR@k /
    NDCG@k over that candidate set. This is the protocol the published HR@10 /
    NDCG@10 numbers use; it is not comparable to full-catalog ranking.
    """
    rng = random.Random(seed)
    user_seen: dict[int, set[int]] = {}
    for user, item in train_pairs:
        user_seen.setdefault(user, set()).add(item)
    for user, item in test_pairs:
        user_seen.setdefault(user, set()).add(item)

    model.eval()
    ndcgs: list[float] = []
    hits: list[float] = []
    precisions: list[float] = []
    for user, item in test_pairs:
        seen = user_seen.get(user, set())
        negatives: list[int] = []
        attempts = 0
        while len(negatives) < num_negatives and attempts < num_negatives * 50:
            candidate = rng.randrange(num_items)
            attempts += 1
            if candidate != item and candidate not in seen:
                negatives.append(candidate)
        candidates = [item] + negatives
        users_tensor = torch.full((len(candidates),), user, dtype=torch.long)
        items_tensor = torch.tensor(candidates, dtype=torch.long)
        with torch.no_grad():
            scores = model(users_tensor, items_tensor)
        order = torch.argsort(scores, descending=True).tolist()
        ranked = [candidates[i] for i in order]
        relevant = {item}
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        hits.append(hit_rate_at_k(ranked, relevant, k))
        precisions.append(precision_at_k(ranked, relevant, k))
    return {
        "ndcg_at_10": mean_metric(ndcgs),
        "hit_rate_at_10": mean_metric(hits),
        "precision_at_10": mean_metric(precisions),
        "protocol": f"sampled_1_vs_{num_negatives}",
    }


def evaluate_ranking(
    model: nn.Module,
    test_pairs: list[tuple[int, int]],
    train_pairs: list[tuple[int, int]],
    num_items: int,
    k: int = 10,
) -> dict[str, float]:
    train_seen: dict[int, set[int]] = {}
    for user, item in train_pairs:
        train_seen.setdefault(user, set()).add(item)
    ndcgs: list[float] = []
    hits: list[float] = []
    precisions: list[float] = []
    for user, item in test_pairs:
        scores = score_all_items(model, user, num_items).clone()
        for seen_item in train_seen.get(user, set()):
            scores[seen_item] = -1e9
        ranked = torch.argsort(scores, descending=True).tolist()
        relevant = {item}
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        hits.append(hit_rate_at_k(ranked, relevant, k))
        precisions.append(precision_at_k(ranked, relevant, k))
    return {
        "ndcg_at_10": mean_metric(ndcgs),
        "hit_rate_at_10": mean_metric(hits),
        "precision_at_10": mean_metric(precisions),
    }

