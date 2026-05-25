"""Phase 6: error analysis of the trained retrieval model.

Produces three artifacts:
  1) Cold-start correlation: Recall@10 bucketed by user train-history size.
  2) Popularity histogram of retrieved items vs ground-truth test items.
  3) Qualitative top-10 vs ground-truth printout for a few example users.

Saves the popularity histogram as `notebooks/popularity_histogram.png`.

Run from project root:
    python src/error_analysis.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import faiss  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import load_model
from index import build_item_index
from dataset import parse_movies

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
NOTEBOOKS = PROJECT_ROOT / "notebooks"


def _retrieve_top_k(model, device, k: int = 10) -> tuple[np.ndarray, list[int]]:
    """For every test user, return top-k items after filtering train positives."""
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    test_df = pd.read_parquet(PROCESSED / "test.parquet")

    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_users = sorted(test_df["user_idx"].unique())

    index, _ = build_item_index(model, device)
    user_idx_t = torch.tensor(test_users, dtype=torch.long, device=device)
    with torch.no_grad():
        user_embs = model.user_tower(user_idx_t).detach().cpu().numpy().astype("float32")

    _, ranked = index.search(user_embs, index.ntotal)

    top_k = np.full((len(test_users), k), -1, dtype=np.int64)
    for i, user in enumerate(test_users):
        seen = train_items.get(user, set())
        kept = []
        for item in ranked[i]:
            if item not in seen:
                kept.append(item)
                if len(kept) >= k:
                    break
        top_k[i] = kept[:k]
    return top_k, test_users


def cold_start_analysis(top_k: np.ndarray, test_users: list[int]) -> None:
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    test_df = pd.read_parquet(PROCESSED / "test.parquet")
    test_items = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    train_counts = train_df.groupby("user_idx").size().to_dict()

    rows = []
    for i, user in enumerate(test_users):
        gt = test_items[user]
        recall = len(set(top_k[i].tolist()) & gt) / len(gt)
        rows.append({"user_idx": user, "n_train": train_counts.get(user, 0), "recall10": recall})
    df = pd.DataFrame(rows)

    bins = [0, 20, 50, 100, 200, 500, 10_000]
    labels = ["1-19", "20-49", "50-99", "100-199", "200-499", "500+"]
    df["bucket"] = pd.cut(df["n_train"], bins=bins, labels=labels, right=False)

    print("\n=== Cold-start analysis: Recall@10 by user train-history size ===")
    print(f"{'bucket':<10}  {'#users':>8}  {'mean Recall@10':>16}")
    for bucket in labels:
        sub = df[df["bucket"] == bucket]
        if len(sub):
            print(f"{bucket:<10}  {len(sub):>8,}  {sub['recall10'].mean():>16.4f}")


def popularity_histogram(top_k: np.ndarray) -> None:
    train_df = pd.read_parquet(PROCESSED / "train.parquet")
    test_df = pd.read_parquet(PROCESSED / "test.parquet")

    # Item popularity = how many users rated it in train. Higher rank = more popular.
    pop_count = train_df.groupby("item_idx").size()
    # Convert to popularity *rank* (0 = most popular). This bounds the x-axis.
    rank = pop_count.rank(method="dense", ascending=False).astype(int) - 1
    item_rank = rank.to_dict()  # item_idx -> rank (0 = most popular)

    retrieved_ranks = [item_rank.get(int(i), -1) for row in top_k for i in row if item_rank.get(int(i), -1) >= 0]
    gt_ranks = [item_rank.get(int(i), -1) for i in test_df["item_idx"] if item_rank.get(int(i), -1) >= 0]

    print("\n=== Popularity bias check (lower rank = more popular) ===")
    print(f"  median rank of RETRIEVED items: {int(np.median(retrieved_ranks))}")
    print(f"  median rank of GROUND-TRUTH items: {int(np.median(gt_ranks))}")
    print(f"  mean rank of retrieved: {np.mean(retrieved_ranks):.1f}")
    print(f"  mean rank of ground-truth: {np.mean(gt_ranks):.1f}")
    print(f"  -> if retrieved median > GT median, the model is UNDER-recommending popular items")

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(max(retrieved_ranks), max(gt_ranks)), 50)
    ax.hist(gt_ranks, bins=bins, alpha=0.55, label="Ground-truth test items", density=True)
    ax.hist(retrieved_ranks, bins=bins, alpha=0.55, label="Retrieved top-10 items", density=True)
    ax.set_xlabel("Popularity rank in train (0 = most popular)")
    ax.set_ylabel("Density")
    ax.set_title("Popularity distribution: retrieved vs ground-truth")
    ax.legend()
    fig.tight_layout()
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    out = NOTEBOOKS / "popularity_histogram.png"
    fig.savefig(out, dpi=120)
    print(f"  saved plot -> {out}")


def qualitative_examples(top_k: np.ndarray, test_users: list[int], n_examples: int = 3) -> None:
    test_df = pd.read_parquet(PROCESSED / "test.parquet")
    meta = torch.load(PROCESSED / "meta.pt", weights_only=False)
    movies = parse_movies(PROJECT_ROOT / "data" / "raw" / "ml-1m" / "movies.dat")
    inv_movie = {idx: mid for mid, idx in meta["movie_to_idx"].items()}
    title_of = dict(zip(movies["movie_id"], movies["title"]))

    def label(item_idx: int) -> str:
        return title_of.get(inv_movie.get(item_idx, -1), f"<item_{item_idx}>")

    test_items = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

    # Pick a strong-recall user, a mid user, and a low-recall user for variety.
    recalls = []
    for i, user in enumerate(test_users):
        gt = set(test_items[user])
        r = len(set(top_k[i].tolist()) & gt) / len(gt)
        recalls.append((r, i, user))
    recalls.sort()
    picks_idx = [
        recalls[-1],                       # best
        recalls[len(recalls) // 2],        # median
        recalls[len([r for r in recalls if r[0] == 0.0]) // 2],  # zero-recall middle
    ]

    print(f"\n=== Qualitative examples ({n_examples} users) ===")
    for tag, (r, i, user) in zip(["BEST recall", "MEDIAN recall", "ZERO recall"], picks_idx):
        print(f"\n[{tag}]  user_idx={user}  Recall@10={r:.2f}  (gt has {len(test_items[user])} items)")
        print("  top-10 recommendations:")
        for rank, item in enumerate(top_k[i], 1):
            marker = "✓" if item in set(test_items[user]) else " "
            print(f"   {rank:>2}. {marker} {label(int(item))}")
        print("  ground-truth (first 10):")
        for rank, item in enumerate(test_items[user][:10], 1):
            print(f"   {rank:>2}.   {label(int(item))}")


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    model, _ = load_model(device)
    top_k, test_users = _retrieve_top_k(model, device, k=10)
    print(f"retrieved top-10 for {len(test_users):,} test users")

    cold_start_analysis(top_k, test_users)
    popularity_histogram(top_k)
    qualitative_examples(top_k, test_users)


if __name__ == "__main__":
    main()
