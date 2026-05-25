"""Phase 1: MovieLens 1M loading, temporal split, and PyTorch Dataset.

Run directly to build processed splits:
    python src/dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-1m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_ratings(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def parse_movies(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def build_mappings(ratings: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Map raw IDs to contiguous 0..N-1 indices. Only IDs present in ratings."""
    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(ratings["movie_id"].unique())
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    return user_to_idx, movie_to_idx


def build_genre_matrix(
    movies: pd.DataFrame, movie_to_idx: dict[int, int]
) -> tuple[np.ndarray, list[str]]:
    """Returns (genre_matrix[num_items, num_genres] as float32, genre_vocab)."""
    # Vocab from movies that appear in our index (some movies have no ratings).
    rated_movies = movies[movies["movie_id"].isin(movie_to_idx)]
    genre_vocab = sorted({g for gs in rated_movies["genres"] for g in gs.split("|")})
    genre_to_col = {g: c for c, g in enumerate(genre_vocab)}

    matrix = np.zeros((len(movie_to_idx), len(genre_vocab)), dtype=np.float32)
    for _, row in rated_movies.iterrows():
        item_idx = movie_to_idx[row["movie_id"]]
        for g in row["genres"].split("|"):
            matrix[item_idx, genre_to_col[g]] = 1.0
    return matrix, genre_vocab


def temporal_split(
    ratings: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user split sorted by timestamp.

    For users with fewer interactions than the split allows, val/test may be empty
    for that user; that's fine — train always gets at least 1.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    grouped = ratings.groupby("user_id", sort=False)

    train_parts, val_parts, test_parts = [], [], []
    for _, group in grouped:
        n = len(group)
        n_train = max(1, int(n * train_frac))
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val = max(0, n - n_train)
            n_test = 0
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train : n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val :])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def remap_ids(
    df: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "user_idx": df["user_id"].map(user_to_idx).astype(np.int64),
            "item_idx": df["movie_id"].map(movie_to_idx).astype(np.int64),
            "rating": df["rating"].astype(np.float32),
            "timestamp": df["timestamp"].astype(np.int64),
        }
    )
    return out


class MovieLensDataset(Dataset):
    """Yields (user_idx, item_idx) pairs as torch.long tensors.

    Phase 2 training uses in-batch negatives, so we only emit positives here.
    """

    def __init__(self, split_df: pd.DataFrame):
        self.users = torch.from_numpy(split_df["user_idx"].to_numpy().copy())
        self.items = torch.from_numpy(split_df["item_idx"].to_numpy().copy())

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx]


def build_and_save() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing raw files...")
    ratings = parse_ratings(RAW_DIR / "ratings.dat")
    movies = parse_movies(RAW_DIR / "movies.dat")
    print(f"  ratings: {len(ratings):,}  movies: {len(movies):,}")

    print("Building ID mappings...")
    user_to_idx, movie_to_idx = build_mappings(ratings)
    print(f"  users: {len(user_to_idx):,}  items: {len(movie_to_idx):,}")

    print("Building genre matrix...")
    genre_matrix, genre_vocab = build_genre_matrix(movies, movie_to_idx)
    print(f"  genre vocab ({len(genre_vocab)}): {genre_vocab}")

    print("Temporal split (per-user 80/10/10)...")
    train_df, val_df, test_df = temporal_split(ratings, 0.8, 0.1)
    print(f"  train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    print("Remapping IDs to indices...")
    train_df = remap_ids(train_df, user_to_idx, movie_to_idx)
    val_df = remap_ids(val_df, user_to_idx, movie_to_idx)
    test_df = remap_ids(test_df, user_to_idx, movie_to_idx)

    print(f"Saving to {PROCESSED_DIR}...")
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    np.save(PROCESSED_DIR / "genre_matrix.npy", genre_matrix)

    meta = {
        "num_users": len(user_to_idx),
        "num_items": len(movie_to_idx),
        "num_genres": len(genre_vocab),
        "genre_vocab": genre_vocab,
        "user_to_idx": {int(k): int(v) for k, v in user_to_idx.items()},
        "movie_to_idx": {int(k): int(v) for k, v in movie_to_idx.items()},
    }
    torch.save(meta, PROCESSED_DIR / "meta.pt")
    print("Done.")


if __name__ == "__main__":
    build_and_save()
