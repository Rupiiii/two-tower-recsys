"""Phase 4: train the BPR ranker on top of the frozen two-tower retrieval model.

Run from project root:
    python src/train_ranker.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

import faiss  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_model
from index import build_item_index
from ranker import Ranker, bpr_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = PROJECT_ROOT / "data" / "checkpoints"


def build_candidate_pool(
    model, device: torch.device, k: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute candidates per user + frozen user/item embeddings.

    Returns:
        candidates: [num_users, k_keep] int64 — top-k FAISS items minus train positives, padded.
        user_embs: [num_users, D] float32
        item_embs: [num_items, D] float32
    """
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    print("Computing frozen embeddings + FAISS index...")
    index, item_embs = build_item_index(model, device)
    num_users = model.user_tower.embedding.num_embeddings

    user_idx_t = torch.arange(num_users, dtype=torch.long, device=device)
    with torch.no_grad():
        user_embs = model.user_tower(user_idx_t).detach().cpu().numpy().astype("float32")

    print(f"Mining top-{k * 2} candidates per user (oversample, then filter)...")
    _, ranked = index.search(user_embs, k * 2)

    candidates = np.full((num_users, k), -1, dtype=np.int64)
    for u in range(num_users):
        seen = train_items.get(u, set())
        kept = [item for item in ranked[u] if item not in seen]
        if len(kept) < k:
            kept = kept + [kept[-1] if kept else 0] * (k - len(kept))
        candidates[u] = kept[:k]

    return candidates, user_embs, item_embs


def train_ranker(
    num_epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    candidate_k: int = 100,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    # Two-tower model runs on MPS for the one-shot embedding extraction, but
    # the ranker itself trains on CPU — sharing an MPS context between a frozen
    # model and an actively-training model deadlocks on this machine, and
    # 24K params runs in seconds on CPU anyway.
    tower_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ranker_device = torch.device("cpu")
    print(f"tower device: {tower_device}  ranker device: {ranker_device}")

    model, cfg = load_model(tower_device)
    model.eval()
    emb_dim = cfg["emb_dim"]

    candidates, user_embs, item_embs = build_candidate_pool(model, tower_device, k=candidate_k)
    print(f"  candidates: {candidates.shape}  user_embs: {user_embs.shape}  item_embs: {item_embs.shape}")

    # Keep frozen embeddings + candidates on CPU; only batch-sized tensors
    # move to device. MPS deadlocks on advanced indexing into device tensors.
    U_cpu = torch.from_numpy(user_embs)
    V_cpu = torch.from_numpy(item_embs)
    C_cpu = torch.from_numpy(candidates)

    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    user_idx_t = torch.from_numpy(train_df["user_idx"].to_numpy().copy()).long()
    pos_idx_t = torch.from_numpy(train_df["item_idx"].to_numpy().copy()).long()
    dataset = TensorDataset(user_idx_t, pos_idx_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"training pairs: {len(dataset):,}  batches/epoch: {len(loader):,}")

    ranker = Ranker(emb_dim=emb_dim, hidden_dim=hidden_dim).to(ranker_device)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=lr)
    print(f"ranker params: {sum(p.numel() for p in ranker.parameters()):,}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        total, n = 0.0, 0
        for u_idx, pos_idx in loader:
            cand_pick = torch.randint(0, candidate_k, (u_idx.size(0),))
            neg_idx = C_cpu[u_idx, cand_pick]

            u = U_cpu[u_idx]
            v_pos = V_cpu[pos_idx]
            v_neg = V_cpu[neg_idx]

            score_pos = ranker(u, v_pos)
            score_neg = ranker(u, v_neg)
            loss = bpr_loss(score_pos, score_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        print(f"epoch {epoch:2d}/{num_epochs}  bpr_loss={total / n:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": ranker.state_dict(),
            "config": {"emb_dim": emb_dim, "hidden_dim": hidden_dim},
        },
        CKPT_DIR / "ranker.pt",
    )
    np.save(PROCESSED / "candidates.npy", candidates)
    print(f"saved ranker -> {CKPT_DIR / 'ranker.pt'}")
    print(f"saved candidates -> {PROCESSED / 'candidates.npy'}")


if __name__ == "__main__":
    train_ranker()
