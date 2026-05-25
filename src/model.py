"""Phase 2: Two-tower model.

UserTower : user_idx -> 64-d L2-normalized vector
ItemTower : item_idx -> 64-d L2-normalized vector (uses learned emb + fixed genre vec)
TwoTowerModel wraps both and returns the pair for InfoNCE training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_users, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_idx)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_genres: int,
        genre_matrix: np.ndarray | torch.Tensor,
        emb_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_items, emb_dim)

        if isinstance(genre_matrix, np.ndarray):
            genre_matrix = torch.from_numpy(genre_matrix.copy())
        self.register_buffer("genre_matrix", genre_matrix.float())

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + num_genres, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(item_idx)
        genres = self.genre_matrix[item_idx]
        x = torch.cat([emb, genres], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)

    def all_item_embeddings(self) -> torch.Tensor:
        """Run every item through the tower at once. Used by Phase 3 (FAISS index)."""
        idx = torch.arange(self.embedding.num_embeddings, device=self.genre_matrix.device)
        return self(idx)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        genre_matrix: np.ndarray | torch.Tensor,
        emb_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, emb_dim, hidden_dim)
        self.item_tower = ItemTower(num_items, num_genres, genre_matrix, emb_dim, hidden_dim)

    def forward(
        self, user_idx: torch.Tensor, item_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_idx), self.item_tower(item_idx)
