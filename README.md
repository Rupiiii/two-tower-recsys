# Two-Tower Recommendation System

A two-stage neural recommender on **MovieLens 1M**: retrieval via a two-tower
model with InfoNCE + in-batch negatives, hard-negative mining from FAISS to
correct popularity sampling bias, and error analysis surfacing concrete
failure modes.

Built in PyTorch (Apple Silicon / MPS) and FAISS (CPU, flat index).

---

## Architecture

```
                                 RETRIEVAL
   user_id ─► UserTower(emb→MLP→L2) ─► user_emb (128-d unit-norm)
                                            │
                                            ▼
                              FAISS IndexFlatIP  ─► top-100 candidates
                                            ▲
   item_idx ─► ItemTower(emb⊕genre→MLP→L2) ─► item_embs (128-d unit-norm)
       │           ▲
       └──────► genre_vec (multi-hot 18-d, fixed)
```

Both towers output unit-norm vectors so the dot product is exactly cosine
similarity — which is what FAISS's `IndexFlatIP` needs to do correct
retrieval.

### Model design choices

| choice                   | value                                         | reason |
|--------------------------|-----------------------------------------------|--------|
| embedding dim            | 128                                           | aligned with `√num_items` heuristic, power of 2 for MPS, ~1 sample/param |
| temperature τ            | 0.07                                          | SimCLR/CLIP convention; bounds InfoNCE logits with unit-norm vectors |
| batch size               | 4096                                          | more in-batch negatives = stronger contrastive signal |
| genre feature            | concatenated multi-hot vector, non-trainable  | fixed side feature for item cold-start |
| L2-normalize             | both tower outputs                            | dot product = cosine; FAISS `IndexFlatIP` compatibility |
| hard-neg mix             | 30% hard / 70% in-batch                       | per Yi et al. 2019 |

---

## Repository layout

```
two_tower_recsys/
├── data/
│   ├── raw/ml-1m/                  # parsed from the GroupLens zip
│   └── processed/                  # train/val/test parquet, meta.pt, hard_negs.npy
├── src/
│   ├── dataset.py                  # parsers, ID mappings, genre matrix, temporal split, Dataset class
│   ├── model.py                    # UserTower, ItemTower, TwoTowerModel
│   ├── train.py                    # InfoNCE training loop with optional hard negatives
│   ├── index.py                    # FAISS IndexFlatIP builder
│   ├── evaluate.py                 # Recall@K and NDCG@K (with re-ranking) eval
│   ├── mine_hard_negatives.py      # per-user hard-negative mining from FAISS
│   ├── ranker.py                   # Phase 4 BPR ranker (see Known Issues)
│   ├── train_ranker.py             # Phase 4 ranker training loop (see Known Issues)
│   └── error_analysis.py           # Phase 6 diagnostics + popularity histogram
├── notebooks/
│   └── popularity_histogram.png    # retrieved vs ground-truth popularity (generated)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.12, faiss-cpu 1.14, on an Apple Silicon
Mac with MPS. CPU fallback works.

### Download the dataset

```bash
cd data/raw && curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip ml-1m.zip
```

### Full pipeline

```bash
python src/dataset.py                # build processed splits
python src/train.py                  # train two-tower with in-batch negatives only
python src/evaluate.py               # Recall@K baseline

python src/mine_hard_negatives.py    # mine 30 hard negatives per user from FAISS
python -c "from src.train import train; train(num_epochs=20, batch_size=4096, \
            emb_dim=128, hidden_dim=256, weight_decay=1e-5, \
            hard_negs_path='data/processed/hard_negs.npy', hard_neg_frac=0.3)"
python src/evaluate.py               # Recall@K with hard-neg model

python src/error_analysis.py         # Phase 6: cold-start, popularity, qualitative
```

---

## Phase-by-phase walkthrough

### Phase 1 — Data ([src/dataset.py](src/dataset.py))

- Parses `ratings.dat` and `movies.dat` from the GroupLens ML-1M release.
- Builds contiguous `user_id → idx` and `movie_id → idx` mappings (raw IDs
  are non-contiguous in the source files).
- Constructs a multi-hot genre vector per item from the pipe-delimited
  `genres` field — vocabulary discovered from data, not hardcoded.
- **Per-user temporal 80/10/10 split** — sort each user's interactions by
  timestamp, slice. The script also verifies the split has zero strict
  temporal violations across users (same-second timestamp ties are
  reported separately as a known property of the MovieLens timestamp
  resolution).

> Random splits leak future data into training and inflate metrics. Temporal
> splitting per-user is non-negotiable for honest recsys evaluation, and
> avoids the cold-user problem that a global temporal cut would create.

### Phase 2 — Two-Tower model ([src/model.py](src/model.py), [src/train.py](src/train.py))

- `UserTower`: `nn.Embedding(num_users, D) → Linear(D→2D) → ReLU → Linear(2D→D) → F.normalize`.
- `ItemTower`: `[nn.Embedding(num_items, D) ‖ genre_vec(G)] → Linear(D+G→2D) → ReLU → Linear(2D→D) → F.normalize`.
- The genre matrix is registered as a non-trainable `buffer` — it moves with
  the model to device and isn't updated by the optimizer.
- Training: in-batch negatives over a `B × B` similarity matrix. The
  diagonal is positives; off-diagonal entries are the `B-1` negatives per
  row, contributed for free by the rest of the batch.
- Loss: InfoNCE / NT-Xent with temperature `τ`. Equivalent to
  `F.cross_entropy((u @ v.T) / τ, arange(B))`.
- Best-val checkpoint is saved per epoch.

### Phase 3 — FAISS retrieval + Recall@K ([src/index.py](src/index.py), [src/evaluate.py](src/evaluate.py))

- Run every item through `ItemTower` to get the item-embedding matrix.
- Build `faiss.IndexFlatIP` (exact inner-product search). Catalogs of a few
  thousand items don't need approximate methods; for production-scale
  catalogs you'd swap in `IndexIVF` or HNSW.
- For each test user, retrieve top-N from FAISS, **filter out items the
  user already saw in training**, count overlap with their held-out test
  items to compute Recall@K.
- **Per-user averaging** — Recall@K is averaged across users, not pooled
  globally. Pooled averaging would let heavy raters dominate the metric.

### Phase 4 — Hard-negative mining ([src/mine_hard_negatives.py](src/mine_hard_negatives.py))

**Motivation: popularity sampling bias.** When items appear as in-batch
negatives proportional to their popularity `P(item)`, the converged model
implicitly estimates `P(item | user) / P(item)` rather than `P(item | user)`.
The `/ P(item)` term subtracts popularity from every score, so popular
items get systematically under-recommended. Yi et al. 2019 derive this for
the YouTube two-tower model.

A simple diagnostic — comparing the in-batch baseline against "just
recommend the K most-rated items in train" — exposes the bias directly,
and motivates the Phase 5 fix.

**Mining recipe**:
- Use the baseline model to build a FAISS index.
- For each user, query top-50, drop their training positives, keep up to 30
  hard negatives. Result: a `[num_users, 30]` int64 matrix.
- A small fraction of very heavy raters end up with fewer distinct hard
  negs after filtering; the script pads with duplicates and reports the
  count.

**Retraining**: same architecture and hyperparameters, but each training
batch additionally samples a chunk of items uniformly from the flattened
hard-neg pool (sized to match the doc's 70/30 in-batch/hard ratio). The
contrastive matrix becomes `B × (B + N_hard)`; positives stay in columns
`0..B-1` so the cross-entropy labels (`arange(B)`) remain unchanged.

**Why this fixes the bias**: hard negatives are sampled **by similarity to
the user**, not by frequency. This decouples the negative-sampling
distribution from `P(item)`, neutralizing the implicit `/ P(item)` term.

### Phase 5 — Error analysis ([src/error_analysis.py](src/error_analysis.py))

Three analyses are run against the final trained model:

1. **Cold-start bucketing** — partition test users by training-history size
   (1–19, 20–49, …, 500+) and report mean Recall@10 per bucket. Surfaces
   the structural and model-capacity factors that determine where the
   model does well vs poorly.
2. **Popularity bias residual** — compare the popularity-rank distribution
   of *retrieved* top-K items against the popularity-rank distribution of
   *ground-truth* test items. A right-skew of the retrieved distribution
   indicates residual popularity bias even after hard-neg training. Output
   saved as a density histogram at `notebooks/popularity_histogram.png`.
3. **Qualitative examples** — for a small set of users (best-recall,
   median-recall, zero-recall), print the model's top-10 recommendations
   alongside their actual held-out ground truth, with movie titles. This
   makes failure modes (era mismatch, taste-cluster confusion, etc.)
   concrete.

---

## Stack

- **Python 3.13** (venv-isolated)
- **PyTorch 2.12** with MPS backend (Apple Silicon)
- **FAISS-CPU 1.14** (`IndexFlatIP` — exact inner-product search; small
  catalog doesn't need approximate nearest-neighbor)
- **pandas / pyarrow** for parquet-backed processed splits
- **matplotlib** for the popularity histogram

---

## References

- Yi, X. et al. (2019). *Sampling-Bias-Corrected Neural Modeling for Large
  Corpus Item Recommendations*. RecSys '19. (Popularity sampling bias and
  the `log P(item)` correction.)
- Chen, T. et al. (2020). *A Simple Framework for Contrastive Learning of
  Visual Representations* (SimCLR). (InfoNCE / NT-Xent loss; τ = 0.07.)
- Rendle, S. et al. (2009). *BPR: Bayesian Personalized Ranking from
  Implicit Feedback*. (Pairwise ranking loss used in the Phase 4 ranker.)
- Covington, P., Adams, J., Sargin, E. (2016). *Deep Neural Networks for
  YouTube Recommendations*. RecSys '16. (Original two-tower retrieval +
  ranking architecture.)
