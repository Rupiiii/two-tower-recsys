"""Phase 3: Recall@K evaluation on the test set.

Run from project root:
    python src/evaluate.py
"""

from __future__ import annotations

import os

# macOS: torch and faiss-cpu both ship libomp; pip installs of either bring
# their own copy. Single-shot inference doesn't race, so this is safe here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import faiss  # noqa: F401  (imported here so the env var is set first)
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from index import build_item_index
from model import TwoTowerModel
from ranker import Ranker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = PROJECT_ROOT / "data" / "checkpoints"


def load_model(device: torch.device) -> tuple[TwoTowerModel, dict]:
    ckpt = torch.load(CKPT_DIR / "best.pt", weights_only=False)
    cfg = ckpt["config"]
    genre = np.load(PROCESSED_DIR / "genre_matrix.npy")
    model = TwoTowerModel(
        num_users=cfg["num_users"],
        num_items=cfg["num_items"],
        num_genres=cfg["num_genres"],
        genre_matrix=genre,
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def evaluate(ks: tuple[int, ...] = (10, 50, 100)) -> dict[int, float]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model, _ = load_model(device)
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print("Building FAISS index...")
    index, _ = build_item_index(model, device)
    print(f"  index size: {index.ntotal}  dim: {index.d}")

    # Ground truth + train-item filter, keyed by user_idx.
    test_items = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users = sorted(test_items.keys())
    print(f"Test users: {len(test_users):,}")

    # Compute all user embeddings in a single forward pass.
    user_idx_t = torch.tensor(test_users, dtype=torch.long, device=device)
    with torch.no_grad():
        user_embs = model.user_tower(user_idx_t).detach().cpu().numpy().astype("float32")

    # Retrieve full ranking (3,706 items — exact search is trivial).
    # Querying num_items means: post-filtering can never run out of candidates.
    num_items = index.ntotal
    _, ranked = index.search(user_embs, num_items)  # ranked[i] = item idxs sorted by score

    max_k = max(ks)
    recalls = {k: [] for k in ks}

    for i, user in enumerate(test_users):
        gt = test_items[user]
        seen = train_items.get(user, set())
        # Drop train items, keep enough for the largest K we need.
        retrieved = []
        for item in ranked[i]:
            if item not in seen:
                retrieved.append(item)
                if len(retrieved) >= max_k:
                    break
        for k in ks:
            top_k = set(retrieved[:k])
            recalls[k].append(len(top_k & gt) / len(gt))

    results = {k: float(np.mean(recalls[k])) for k in ks}
    print("\n=== Recall@K (averaged over test users) ===")
    for k in ks:
        print(f"  Recall@{k:>3} = {results[k]:.4f}  ({results[k] * 100:.2f}%)")
    return results


def _dcg(rel: list[int]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(retrieved: list[int], ground_truth: set[int], k: int) -> float:
    rel = [1 if item in ground_truth else 0 for item in retrieved[:k]]
    n_hits = min(len(ground_truth), k)
    if n_hits == 0:
        return 0.0
    idcg = _dcg([1] * n_hits)
    return _dcg(rel) / idcg


def evaluate_ndcg(candidate_k: int = 100, ndcg_k: int = 10) -> dict[str, float]:
    """NDCG@k before vs after re-ranking the top-`candidate_k` from FAISS."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model, cfg = load_model(device)
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print("Building FAISS index + frozen embeddings...")
    index, item_embs_np = build_item_index(model, device)
    item_embs = torch.from_numpy(item_embs_np).to(device)

    # User embeddings for the test users.
    test_items = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users = sorted(test_items.keys())
    user_idx_t = torch.tensor(test_users, dtype=torch.long, device=device)
    with torch.no_grad():
        user_embs = model.user_tower(user_idx_t)
        user_embs_np = user_embs.detach().cpu().numpy().astype("float32")

    print(f"Retrieving top-{candidate_k} per user...")
    _, ranked = index.search(user_embs_np, index.ntotal)

    # Build filtered top-K candidate list per user.
    candidates = np.full((len(test_users), candidate_k), -1, dtype=np.int64)
    for i, user in enumerate(test_users):
        seen = train_items.get(user, set())
        kept = []
        for item in ranked[i]:
            if item not in seen:
                kept.append(item)
                if len(kept) >= candidate_k:
                    break
        candidates[i] = kept[:candidate_k]

    # Load ranker.
    print("Loading ranker...")
    rckpt = torch.load(CKPT_DIR / "ranker.pt", weights_only=False)
    ranker = Ranker(emb_dim=cfg["emb_dim"], hidden_dim=rckpt["config"]["hidden_dim"]).to(device)
    ranker.load_state_dict(rckpt["state_dict"])
    ranker.eval()

    # Score every (user, candidate) pair with the ranker.
    print(f"Re-ranking {len(test_users):,} × {candidate_k} candidates...")
    cand_t = torch.from_numpy(candidates).to(device)
    with torch.no_grad():
        # [U, K, D] for items and [U, 1, D] for user broadcast.
        u_b = user_embs.unsqueeze(1).expand(-1, candidate_k, -1)
        v_b = item_embs[cand_t]
        scores = ranker(u_b.reshape(-1, u_b.size(-1)), v_b.reshape(-1, v_b.size(-1)))
        scores = scores.view(len(test_users), candidate_k)
    new_order = torch.argsort(scores, dim=-1, descending=True).cpu().numpy()

    # Compute NDCG@k before and after.
    ndcg_before, ndcg_after = [], []
    for i, user in enumerate(test_users):
        gt = test_items[user]
        original = candidates[i].tolist()
        reranked = candidates[i][new_order[i]].tolist()
        ndcg_before.append(ndcg_at_k(original, gt, ndcg_k))
        ndcg_after.append(ndcg_at_k(reranked, gt, ndcg_k))

    before_mean = float(np.mean(ndcg_before))
    after_mean = float(np.mean(ndcg_after))
    delta_pct = (after_mean - before_mean) / before_mean * 100 if before_mean > 0 else 0.0

    print(f"\n=== NDCG@{ndcg_k} (top-{candidate_k} candidate pool) ===")
    print(f"  before re-ranking (FAISS order): {before_mean:.4f}")
    print(f"  after  re-ranking (ranker)     : {after_mean:.4f}")
    print(f"  improvement                     : {delta_pct:+.2f}%")
    return {"before": before_mean, "after": after_mean, "delta_pct": delta_pct}


if __name__ == "__main__":
    evaluate()
    evaluate_ndcg()
