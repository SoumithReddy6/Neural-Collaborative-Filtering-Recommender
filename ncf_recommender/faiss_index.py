from __future__ import annotations

import time
from dataclasses import dataclass
import os

import numpy as np


@dataclass
class SearchResult:
    item_indices: list[int]
    scores: list[float]
    latency_ms: float


class ItemVectorIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype("float32")
        self._faiss_index = None
        if os.getenv("RECOMMENDER_USE_FAISS", "0") != "1":
            return
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss_index = index
        except Exception:
            self._faiss_index = None

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> SearchResult:
        start = time.perf_counter()
        query = query_vector.astype("float32").reshape(1, -1)
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, top_k)
            result = SearchResult(indices[0].tolist(), scores[0].tolist(), (time.perf_counter() - start) * 1000)
            return result
        scores = self.vectors @ query.reshape(-1)
        indices = np.argsort(-scores)[:top_k]
        return SearchResult(indices.tolist(), scores[indices].astype(float).tolist(), (time.perf_counter() - start) * 1000)
