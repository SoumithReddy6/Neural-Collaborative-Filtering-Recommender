# MLflow Experiment Plan

The experiment manifest covers 162 runs:

- Models: `ncf`, `mf`
- Learning rates: `0.0005`, `0.001`, `0.003`
- Embedding dimensions: `32`, `64`, `128`
- Negatives per positive: `2`, `4`, `8`
- Seeds: `13`, `21`, `42`

Primary tracked metrics:

- `ndcg_at_10`
- `hit_rate_at_10`
- `precision_at_10`
- `relative_ndcg_lift`
- `train_bpr_loss`
- `p95_inference_latency_ms`
- `ctr_lift`
- `ab_test_p_value`

Generate the manifest:

```bash
python3 scripts/generate_mlflow_runs.py --output artifacts/experiment_manifest.csv
```
