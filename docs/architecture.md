# Architecture

## System Goal

The project demonstrates a production-shaped recommendation pipeline: ingest implicit feedback, train a neural collaborative filtering model, compare it against a matrix-factorization baseline, export item vectors for retrieval, and serve recommendations through FastAPI.

## Components

- Data layer: normalizes MovieLens/Amazon/Yelp-style interactions into `user_id,item_id,rating,timestamp,clicked,session_id,position`.
- Modeling layer: trains NCF and MF models with BPR loss and Adam.
- Evaluation layer: computes NDCG@10, HitRate@10, Precision@10, and relative lift.
- Retrieval layer: builds a FAISS-compatible item-vector index with a NumPy fallback for local validation.
- Serving layer: exposes `/recommend` through FastAPI using the exported vector index.
- Experiment layer: documents 54 MLflow-ready runs across model and training parameters.
- Analytics layer: computes CTR, session depth, and A/B simulation results.

## Production Extension

For a full production build, the next steps are streaming feature ingestion, scheduled offline retraining, online feature store integration, candidate generation plus ranking separation, and guardrails for popularity bias and cold-start users.

