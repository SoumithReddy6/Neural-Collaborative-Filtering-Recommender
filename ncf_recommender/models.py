from __future__ import annotations

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.05)
        nn.init.normal_(self.item_embeddings.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embeddings(users) * self.item_embeddings(items)).sum(dim=-1)


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [64, 32, 16]
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.user_embeddings.weight, std=0.05)
        nn.init.normal_(self.item_embeddings.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.user_embeddings(users), self.item_embeddings(items)], dim=-1)
        return self.mlp(features).squeeze(-1)

    def item_vectors(self) -> torch.Tensor:
        return self.item_embeddings.weight.detach()


def score_all_items(model: nn.Module, user_idx: int, num_items: int) -> torch.Tensor:
    model.eval()
    users = torch.full((num_items,), user_idx, dtype=torch.long)
    items = torch.arange(num_items, dtype=torch.long)
    with torch.no_grad():
        return model(users, items)

