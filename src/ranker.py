"""Phase 4: pairwise BPR ranker over [u ‖ v ‖ u⊙v] features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ranker(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([u, v, u * v], dim=-1)
        return self.mlp(x).squeeze(-1)


def bpr_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(score_pos - score_neg).mean()
