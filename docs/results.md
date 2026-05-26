# Results

Real training run on MovieLens, executed locally on CPU.

- **Models:** Neural Collaborative Filtering (MLP tower) vs Matrix Factorization baseline
- **Loss:** BPR (pairwise ranking), 4 negatives per positive
- **Data:** MovieLens `ml-latest-small` (~100K ratings, 610 users, ~9.7K movies), leave-one-out split
- **Training:** 20 epochs, embedding dim 32, Adam lr 1e-3
- **Reproduce:**
  ```bash
  python3 scripts/prepare_movielens.py --input <ratings.csv> --output data/processed/interactions.csv
  python3 scripts/evaluate_models.py --interactions data/processed/interactions.csv --epochs 20 --protocol sampled
  ```

## Sampled protocol (1 positive vs 99 sampled negatives — He et al. 2017)

| Metric | NCF | Matrix Factorization |
| --- | ---: | ---: |
| HR@10 | 0.596 | **0.640** |
| NDCG@10 | 0.341 | **0.391** |
| Precision@10 | 0.060 | 0.064 |

These fall in the published-benchmark range for MovieLens with this protocol.

## Full-catalog protocol (rank against all ~9.7K items)

| Metric | NCF | Matrix Factorization |
| --- | ---: | ---: |
| HR@10 | 0.016 | 0.030 |
| NDCG@10 | 0.0066 | 0.014 |

Same models, ~20x lower — full-catalog ranking is much harder. This is why the
evaluation protocol must always be stated alongside HR@10 / NDCG@10.

## Finding

The well-tuned Matrix Factorization baseline outperforms NCF under both
protocols. This mirrors Dacrema et al. (2019), "Are We Really Making Much
Progress? A Worrying Analysis of Recent Neural Recommendation Approaches," which
found strong MF baselines frequently match or beat neural recommenders on these
benchmarks. The contribution of this project is the rigorous, protocol-correct
comparison — not a claim that the neural model wins.

## Honest scope

- The A/B-test module (`scripts/run_ab_test.py`) is a two-proportion z-test on
  **synthetic** CTRs (`simulate_ab_test`), included to demonstrate the statistics.
  It is not a real online experiment.
- Next steps to potentially close the NCF–MF gap: harder negative sampling,
  longer training, and the NeuMF variant that fuses MF and MLP branches.
