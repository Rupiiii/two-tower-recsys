"""Phase 3: build a FAISS inner-product index over all item embeddings."""

from __future__ import annotations

import faiss
import numpy as np
import torch


def build_item_index(model, device) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """Run every item through ItemTower once, build an exact IP index.

    Returns (index, item_embs). item_embs is the contiguous-indexed array
    used by callers that need to inspect a specific item's vector.
    """
    model.eval()
    with torch.no_grad():
        item_embs = model.item_tower.all_item_embeddings()
        item_embs = item_embs.detach().cpu().numpy().astype("float32")

    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    return index, item_embs
