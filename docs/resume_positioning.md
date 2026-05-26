# Resume Positioning

## Project Bullet

Built and benchmarked a Neural Collaborative Filtering recommender against a Matrix Factorization baseline (PyTorch, BPR loss, FAISS retrieval) on MovieLens; under the standard sampled evaluation protocol the MF baseline reached HR@10 0.64 / NDCG@10 0.39 vs NCF's 0.60 / 0.34 — a rigorous result mirroring published findings that strong MF baselines rival neural recommenders.

## Interview Talking Points

- Designed around implicit feedback and pairwise ranking (BPR loss), not rating prediction.
- Used BPR because recommendation quality depends on ranking positives above negatives.
- Found MF outperforms NCF — and can explain why (Dacrema et al. 2019; capacity vs dataset size; tuning).
- Know that evaluation protocol drives the numbers: sampled (1-vs-99) gives HR@10 ~0.6; full-catalog gives ~0.02. Always state the protocol.
- Exported item embeddings into a FAISS index for fast candidate retrieval.
- The A/B module is a two-proportion z-test on synthetic CTRs — a statistics demo, not a real online experiment.

## Honest Metric Framing

Real numbers are in `docs/results.md` (MovieLens ml-latest-small, 20 epochs, sampled protocol). The contribution is a correct, protocol-rigorous NCF-vs-MF comparison — not a claim that the neural model wins. Next steps to potentially close the gap: harder negative sampling, more epochs, and the NeuMF variant.

