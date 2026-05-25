# Two-Tower Recommendation System

A two-stage neural recommender on **MovieLens 1M**: retrieval via a two-tower
model with InfoNCE + in-batch negatives, hard-negative mining from FAISS to
correct popularity sampling bias, and error analysis surfacing concrete
failure modes.

Built in PyTorch (Apple Silicon / MPS) and FAISS (CPU, flat index).

---

## Headline results

| metric        | random | popularity baseline | two-tower (in-batch) | **two-tower + hard negatives** |
|---------------|-------:|--------------------:|---------------------:|-------------------------------:|
| Recall@10     |  0.27% |               3.56% |                2.13% |                      **3.26%** |
| Recall@50     |  1.35% |              13.35% |                9.64% |                     **13.22%** |
| Recall@100    |  2.70% |              21.74% |               17.17% |                     **22.65%** |

**Key finding:** the in-batch-negatives baseline initially *lost* to a
popularity baseline (2.13% vs 3.56% Recall@10) — a textbook case of
**popularity sampling bias** described in Yi et al. 2019 (the YouTube
two-tower paper). Mining hard negatives from the model's own FAISS index
and retraining with a 70/30 in-batch/hard mix lifted Recall@10 by **+53%**
(2.13% → 3.26%) and Recall@100 by **+32%** (17.17% → 22.65%), letting the
model beat the popularity baseline at K=100.

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
similarity, which is what FAISS's `IndexFlatIP` needs to do correct retrieval.

### Model design choices

| choice                   | value                                         | reason |
|--------------------------|-----------------------------------------------|--------|
| embedding dim            | 128                                           | aligned with `√num_items` heuristic, power of 2 for MPS, ~1 sample/param |
| temperature τ            | 0.07                                          | SimCLR/CLIP convention; bounds InfoNCE logits with unit-norm vectors |
| batch size               | 4096                                          | more in-batch negatives = stronger contrastive signal |
| genre feature            | concatenated multi-hot vector, non-trainable  | fixed side feature for item cold-start |
| L2-normalize             | both tower outputs                            | dot product = cosine; FAISS `IndexFlatIP` compatibility |
| hard-neg mix             | 30% hard / 70% in-batch                       | per the Yi 2019 recipe                                                |

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
python src/evaluate.py               # Recall@K baseline numbers

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

- Parses `ratings.dat` (1,000,209 interactions), `movies.dat` (3,883 movies).
- Builds contiguous `user_id → idx` and `movie_id → idx` mappings (raw IDs
  are non-contiguous; max movie ID is 3952 but only 3,706 movies were
  actually rated).
- Constructs an 18-genre multi-hot vector per item from the pipe-delimited
  `genres` field (vocabulary discovered from data, not hardcoded).
- **Per-user temporal 80/10/10 split** (sort each user's interactions by
  timestamp, slice). Verified zero strict temporal violations across all
  6,040 users; 130 boundary cases are same-second ties from MovieLens's
  1-second timestamp resolution.

> Random splits leak future data into training and inflate metrics. Temporal
> split is non-negotiable for any honest recsys evaluation.

### Phase 2 — Two-Tower model ([src/model.py](src/model.py), [src/train.py](src/train.py))

- `UserTower`: `nn.Embedding(num_users, 128) → Linear(128→256) → ReLU → Linear(256→128) → F.normalize`.
- `ItemTower`: `[nn.Embedding(num_items, 128) ‖ genre_vec(18)] → Linear(146→256) → ReLU → Linear(256→128) → F.normalize`.
- Genre vector registered as a non-trainable buffer, moves with the model.
- Training: in-batch negatives over a `B × B` similarity matrix (`B=4096`).
- Loss: InfoNCE / NT-Xent with τ = 0.07 — equivalent to `F.cross_entropy(logits, arange(B))`.
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5, 20 epochs, best-val checkpoint.

Final in-batch baseline: **train loss 7.55 / val loss 8.21 (random baseline at B=4096 is `ln 4096 ≈ 8.32`)**.

### Phase 3 — FAISS retrieval + Recall@K ([src/index.py](src/index.py), [src/evaluate.py](src/evaluate.py))

- Run every item through `ItemTower`, build `faiss.IndexFlatIP` (exact inner
  product; trivial for 3,706 items).
- For each test user, retrieve top-N from FAISS, **exclude training
  positives**, count overlap with their test items.
- **Per-user averaging** of Recall@K (not pooled globally — heavy raters
  would otherwise dominate the metric).

Result: **Recall@10 = 2.13%** with the tuned in-batch model.

### Phase 5 — Hard-negative mining ([src/mine_hard_negatives.py](src/mine_hard_negatives.py))

**Motivation**: The popularity baseline beats the in-batch baseline at K=10
(3.56% vs 2.13%) — a known failure mode of in-batch contrastive training
called *popularity sampling bias* (Yi et al. 2019, "Sampling-Bias-Corrected
Neural Modeling for Large Corpus Item Recommendations").

The math: when items appear as in-batch negatives proportional to their
popularity `P(item)`, the converged model estimates `P(item | user) / P(item)`
rather than `P(item | user)`. The `/ P(item)` term subtracts popularity from
every score, so popular items get systematically under-recommended.

**Mining recipe** (per the project spec):
- Use the in-batch baseline to build a FAISS index.
- For each user, query top-50, drop their training positives → keep up to 30
  hard negatives per user. Result: `[6040, 30]` int64 matrix.
- 5/6040 users had < 10 distinct hard negs (very heavy raters whose train
  history consumed most of the top-50); padded with duplicates. <1% bias.

**Retraining**: same architecture, same hyperparams, but each batch
additionally samples 1,755 items uniformly from the flattened hard-neg pool
(~30% of negatives). The contrastive matrix becomes `4096 × 5851`.

**Result**: Recall@10 = **3.26%** (+53% vs baseline), beating popularity at
K=50 (tie) and K=100 (22.65% vs 21.74%).

Why hard negatives fix the bias: hard negatives are sampled by similarity
to the user, not by frequency. This decouples the negative-sampling
distribution from `P(item)`, neutralizing the implicit `/ P(item)` term.

### Phase 6 — Error analysis ([src/error_analysis.py](src/error_analysis.py))

**Cold-start bucket analysis** — Recall@10 by user train-history size:

| train history | #users | mean Recall@10 |
|---------------|-------:|---------------:|
| 1-19          |    416 |         0.0681 |
| 20-49         |  1,789 |         0.0526 |
| 50-99         |  1,382 |         0.0291 |
| 100-199       |  1,228 |         0.0182 |
| 200-499       |    991 |         0.0101 |
| 500+          |    234 |         0.0074 |

Recall@10 *decreases* monotonically with train-history size. Two reasons
combined:
- **Metric-structural** — heavy raters have ~60 test items, so Recall@10 is
  capped at `10/60 ≈ 17%`. Light raters with 2 test items can hit 100%.
- **Model-structural** — heavy raters have diverse tastes that don't
  compress well into a single 128-d point.

**Popularity bias residual**:

| | retrieved items | ground-truth items |
|---|---:|---:|
| median popularity rank in train | **525** | **456** |
| mean popularity rank            | 477.5   | 433.6  |

The model retrieves items that are *less popular* than what users actually
liked — direct evidence that some popularity bias remains even after hard
negatives. See [notebooks/popularity_histogram.png](notebooks/popularity_histogram.png)
for the distribution overlap.

**Qualitative example** (one of three from the script):

> User 3939 (gt=50 items, era: 1950s–1970s classic cinema):
> ground truth includes *Kramer Vs. Kramer (1979)*, *Doctor Zhivago (1965)*,
> *Streetcar Named Desire (1951)*, *Midnight Cowboy (1969)*.
>
> Top-10 from our model: *Hamlet (1990)*, *Primary Colors (1998)*,
> *Spanish Prisoner (1997)*, *Six Degrees of Separation (1993)* — late-90s
> indie films.
>
> **Failure mode**: the model picked up on "drama" but missed the user's
> strong era preference. Genre features alone don't disambiguate decade.

---

## Phase 4 — Pairwise BPR ranker (known issue)

**Status**: implemented but not benchmarked due to an environment issue.

Files: [src/ranker.py](src/ranker.py), [src/train_ranker.py](src/train_ranker.py).
The `evaluate.py` script also contains an `evaluate_ndcg()` function that
computes NDCG@10 before and after re-ranking.

Design: a small MLP over `[user_emb ‖ item_emb ‖ user_emb ⊙ item_emb]`
features (384-d → 64-d → scalar), trained with BPR loss
`L = -log σ(score_pos - score_neg)` where negatives are sampled from each
user's FAISS top-100 candidates.

**Why it didn't run**: training the ranker on the same MPS context as the
frozen two-tower model deadlocks on Python 3.13 + torch 2.12 + faiss-cpu
on Apple Silicon. Both pure-MPS and CPU-side-indexing variants hang at 0%
CPU with no progress. The deadlock survives moving the ranker entirely to
CPU while the towers stay on MPS, suggesting a shared OpenMP / MPS
synchronization issue rather than a code bug.

**Workaround paths** (untested):
- Rebuild the venv with Python 3.12 — torch's MPS path is better-tested
  on 3.12 than on 3.13 as of this writing.
- Move both towers + ranker to CPU for the ranker-training phase only.

The retrieval-side narrative stands on its own; ranking would have been
the polish step on top of an already-strong retriever.

---

## Stack

- **Python 3.13** (venv-isolated)
- **PyTorch 2.12** with MPS backend (Apple Silicon)
- **FAISS-CPU 1.14** (`IndexFlatIP` — exact inner-product search; 3,706 items
  doesn't need ANN)
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
