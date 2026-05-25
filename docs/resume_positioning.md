# Resume Positioning

## Project Bullet

Built a Neural Collaborative Filtering recommendation system with PyTorch embeddings, BPR loss, FAISS retrieval, FastAPI inference, MLflow experiment tracking, and A/B testing simulation against a matrix-factorization baseline.

## Interview Talking Points

- Designed the model around implicit feedback and pairwise ranking, not just rating prediction.
- Used BPR loss because recommendation quality depends on ranking positive items above negatives.
- Compared NCF against matrix factorization to show measurable model lift.
- Exported item embeddings into a FAISS-compatible index for fast candidate retrieval.
- Added A/B simulation to connect offline ranking metrics to product CTR impact.
- Measured CTR and session depth to show understanding of user interaction patterns.

## Honest Metric Framing

The repository includes a deterministic smoke fixture and the full training/evaluation harness. The target NDCG@10 and lift numbers should be reported after running the documented scripts on MovieLens-25M or another full public dataset.

