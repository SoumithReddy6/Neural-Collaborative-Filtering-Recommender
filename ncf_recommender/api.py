from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ncf_recommender.faiss_index import ItemVectorIndex


class RecommendRequest(BaseModel):
    user_vector: list[float] = Field(..., min_length=2)
    top_k: int = Field(10, ge=1, le=100)


class RecommendResponse(BaseModel):
    item_ids: list[str]
    scores: list[float]
    latency_ms: float


def load_index(path: str | Path) -> tuple[ItemVectorIndex, dict[int, str]]:
    artifact = np.load(path, allow_pickle=True)
    vectors = artifact["item_vectors"]
    item_ids = artifact["item_ids"].tolist()
    return ItemVectorIndex(vectors), {idx: item_id for idx, item_id in enumerate(item_ids)}


app = FastAPI(title="Neural Collaborative Filtering Recommender", version="0.1.0")
INDEX_PATH = Path(os.getenv("RECOMMENDER_INDEX_PATH", "artifacts/item_vectors.npz"))
_index: ItemVectorIndex | None = None
_items: dict[int, str] = {}


@app.on_event("startup")
def startup() -> None:
    global _index, _items
    if INDEX_PATH.exists():
        _index, _items = load_index(INDEX_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "index_loaded": _index is not None}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    if _index is None:
        raise HTTPException(status_code=503, detail="Recommendation index is not loaded. Run scripts/run_smoke_demo.py first.")
    result = _index.search(np.array(request.user_vector, dtype="float32"), request.top_k)
    item_ids = [_items.get(idx, str(idx)) for idx in result.item_indices]
    return RecommendResponse(item_ids=item_ids, scores=result.scores, latency_ms=result.latency_ms)

