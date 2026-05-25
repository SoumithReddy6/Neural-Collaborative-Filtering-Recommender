import numpy as np

from ncf_recommender.faiss_index import ItemVectorIndex


def test_vector_index_returns_nearest_items():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype="float32")
    result = ItemVectorIndex(vectors).search(np.array([1.0, 0.0], dtype="float32"), top_k=2)
    assert result.item_indices[0] == 0
    assert result.latency_ms >= 0

