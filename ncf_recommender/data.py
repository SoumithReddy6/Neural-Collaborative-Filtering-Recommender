from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    rating: float
    timestamp: int
    clicked: int
    session_id: str
    position: int


@dataclass
class EncodedInteractions:
    users: list[int]
    items: list[int]
    labels: list[int]
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]


def load_interactions(path: str | Path) -> list[Interaction]:
    rows: list[Interaction] = []
    with Path(path).open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Interaction(
                    user_id=row["user_id"],
                    item_id=row["item_id"],
                    rating=float(row["rating"]),
                    timestamp=int(row["timestamp"]),
                    clicked=int(row.get("clicked", "1")),
                    session_id=row.get("session_id", ""),
                    position=int(row.get("position", "1")),
                )
            )
    return rows


def encode_interactions(interactions: Iterable[Interaction], min_rating_positive: float = 4.0) -> EncodedInteractions:
    interactions = list(interactions)
    user_ids = sorted({row.user_id for row in interactions})
    item_ids = sorted({row.item_id for row in interactions})
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    return EncodedInteractions(
        users=[user_to_idx[row.user_id] for row in interactions],
        items=[item_to_idx[row.item_id] for row in interactions],
        labels=[1 if row.rating >= min_rating_positive or row.clicked else 0 for row in interactions],
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )


def leave_one_out_split(encoded: EncodedInteractions) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    by_user: dict[int, list[int]] = {}
    for user, item, label in zip(encoded.users, encoded.items, encoded.labels):
        if label:
            by_user.setdefault(user, []).append(item)
    train: list[tuple[int, int]] = []
    test: list[tuple[int, int]] = []
    for user, items in by_user.items():
        if len(items) == 1:
            train.append((user, items[0]))
            test.append((user, items[0]))
            continue
        train.extend((user, item) for item in items[:-1])
        test.append((user, items[-1]))
    return train, test


def sample_bpr_triples(
    positives: list[tuple[int, int]],
    num_items: int,
    negatives_per_positive: int = 3,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    user_pos: dict[int, set[int]] = {}
    for user, item in positives:
        user_pos.setdefault(user, set()).add(item)
    triples: list[tuple[int, int, int]] = []
    for user, pos_item in positives:
        for _ in range(negatives_per_positive):
            neg_item = rng.randrange(num_items)
            while neg_item in user_pos[user]:
                neg_item = rng.randrange(num_items)
            triples.append((user, pos_item, neg_item))
    return triples

