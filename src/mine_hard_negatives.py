"""Phase 5: mine hard negatives from FAISS-retrieved candidates per user.

For each user, query top-K_RETRIEVE from FAISS, drop their training positives,
keep the first K_KEEP that remain. Saves a [num_users, K_KEEP] int64 matrix.

Run from project root:
    python src/mine_hard_negatives.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import faiss  # noqa: F401
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_model
from index import build_item_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"


def mine(k_retrieve: int = 50, k_keep: int = 30) -> Path:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model, cfg = load_model(device)
    train_df = pd.read_parquet(PROCESSED / "train.parquet")

    print("Building FAISS index from in-batch baseline checkpoint...")
    index, _ = build_item_index(model, device)

    num_users = cfg["num_users"]
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    print(f"Mining top-{k_retrieve} candidates per user...")
    user_idx_t = torch.arange(num_users, dtype=torch.long, device=device)
    with torch.no_grad():
        user_embs = model.user_tower(user_idx_t).detach().cpu().numpy().astype("float32")
    _, ranked = index.search(user_embs, k_retrieve)

    print(f"Filtering out train positives, keeping {k_keep} hard negs per user...")
    hard_negs = np.full((num_users, k_keep), -1, dtype=np.int64)
    short_users = 0
    for u in range(num_users):
        seen = train_items.get(u, set())
        kept = [item for item in ranked[u] if item not in seen]
        if len(kept) < k_keep:
            short_users += 1
            # Pad by repeating the last available item; rare with k_retrieve=50.
            kept = kept + [kept[-1] if kept else 0] * (k_keep - len(kept))
        hard_negs[u] = kept[:k_keep]

    print(f"  hard_negs shape: {hard_negs.shape}")
    print(f"  users short on hard negs after filtering: {short_users}")
    print(f"  example user 0 hard negs (first 5): {hard_negs[0][:5].tolist()}")

    out_path = PROCESSED / "hard_negs.npy"
    np.save(out_path, hard_negs)
    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    mine()
