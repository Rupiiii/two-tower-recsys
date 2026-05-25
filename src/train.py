"""Phase 2: train the two-tower model with InfoNCE + in-batch negatives.

Run from project root:
    python src/train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import MovieLensDataset
from model import TwoTowerModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = PROJECT_ROOT / "data" / "checkpoints"


def info_nce_loss(u: torch.Tensor, v: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """In-batch contrastive loss.

    u, v: [B, D] L2-normalized. Diagonal of u @ v.T are positives;
    off-diagonal entries are the (B-1) in-batch negatives per row.
    """
    logits = (u @ v.T) / temperature             # [B, B]
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    temperature: float,
    train: bool,
    hard_neg_pool: torch.Tensor | None = None,
    n_hard: int = 0,
) -> float:
    model.train(train)
    total, n_batches = 0.0, 0
    use_hard = train and hard_neg_pool is not None and n_hard > 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for u_idx, i_idx in loader:
            u_idx, i_idx = u_idx.to(device), i_idx.to(device)
            if use_hard:
                idx = torch.randint(0, len(hard_neg_pool), (n_hard,), device=device)
                items = torch.cat([i_idx, hard_neg_pool[idx]])
            else:
                items = i_idx
            u = model.user_tower(u_idx)
            v = model.item_tower(items)
            loss = info_nce_loss(u, v, temperature=temperature)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)


def train(
    num_epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    emb_dim: int = 64,
    hidden_dim: int = 128,
    temperature: float = 0.07,
    seed: int = 42,
    hard_negs_path: str | None = None,
    hard_neg_frac: float = 0.3,
) -> None:
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    meta = torch.load(PROCESSED_DIR / "meta.pt", weights_only=False)
    genre_matrix = np.load(PROCESSED_DIR / "genre_matrix.npy")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    print(f"train: {len(train_df):,}  val: {len(val_df):,}")

    hard_neg_pool = None
    n_hard = 0
    if hard_negs_path:
        flat = np.load(hard_negs_path).flatten()
        hard_neg_pool = torch.from_numpy(flat).long().to(device)
        n_hard = round(hard_neg_frac * (batch_size - 1) / (1 - hard_neg_frac))
        print(f"hard negs: pool={len(hard_neg_pool):,}  sampling {n_hard}/batch ({hard_neg_frac:.0%} of negs)")

    train_loader = DataLoader(
        MovieLensDataset(train_df), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        MovieLensDataset(val_df), batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = TwoTowerModel(
        num_users=meta["num_users"],
        num_items=meta["num_items"],
        num_genres=meta["num_genres"],
        genre_matrix=genre_matrix,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(
            model, train_loader, optimizer, device, temperature, train=True,
            hard_neg_pool=hard_neg_pool, n_hard=n_hard,
        )
        val_loss = _run_epoch(model, val_loader, optimizer, device, temperature, train=False)
        dt = time.time() - t0

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "emb_dim": emb_dim,
                        "hidden_dim": hidden_dim,
                        "temperature": temperature,
                        "num_users": meta["num_users"],
                        "num_items": meta["num_items"],
                        "num_genres": meta["num_genres"],
                    },
                },
                CKPT_DIR / "best.pt",
            )
            flag = "  <- best"
        print(f"epoch {epoch:2d}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  ({dt:.1f}s){flag}")

    print(f"\nbest val loss: {best_val:.4f}  saved to {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    train()
