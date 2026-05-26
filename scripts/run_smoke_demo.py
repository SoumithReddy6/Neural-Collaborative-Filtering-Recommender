#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ncf_recommender.ab_testing import simulate_ab_test
from ncf_recommender.data import encode_interactions, leave_one_out_split, load_interactions, sample_bpr_triples
from ncf_recommender.stats import interaction_summary
from ncf_recommender.training import build_model, evaluate_ranking, train_bpr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default="data/samples/interactions.csv")
    parser.add_argument("--output", default="artifacts/smoke_metrics.json")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    interactions = load_interactions(args.interactions)
    encoded = encode_interactions(interactions)
    train_pairs, test_pairs = leave_one_out_split(encoded)
    triples = sample_bpr_triples(train_pairs, len(encoded.item_to_idx), negatives_per_positive=4)

    ncf = build_model("ncf", len(encoded.user_to_idx), len(encoded.item_to_idx), embedding_dim=16)
    mf = build_model("mf", len(encoded.user_to_idx), len(encoded.item_to_idx), embedding_dim=16)
    ncf_result = train_bpr(ncf, triples, epochs=args.epochs, learning_rate=0.005)
    mf_result = train_bpr(mf, triples, epochs=args.epochs, learning_rate=0.005)

    ncf_metrics = evaluate_ranking(ncf_result.model, test_pairs, train_pairs, len(encoded.item_to_idx), k=10)
    mf_metrics = evaluate_ranking(mf_result.model, test_pairs, train_pairs, len(encoded.item_to_idx), k=10)
    improvement = (
        (ncf_metrics["ndcg_at_10"] - mf_metrics["ndcg_at_10"]) / mf_metrics["ndcg_at_10"]
        if mf_metrics["ndcg_at_10"]
        else 0.0
    )
    stats = interaction_summary(interactions)
    ab_result = simulate_ab_test(baseline_ctr=0.122, treatment_ctr=0.146, users_per_arm=5000)

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    item_vectors = ncf.item_vectors().numpy().astype("float32")
    np.savez(
        artifact_dir / "item_vectors.npz",
        item_vectors=item_vectors,
        item_ids=np.array([encoded.idx_to_item[idx] for idx in range(len(encoded.idx_to_item))]),
    )
    torch.save(ncf.state_dict(), artifact_dir / "ncf_model.pt")

    metrics = {
        "ncf": ncf_metrics,
        "matrix_factorization": mf_metrics,
        "relative_ndcg_lift": improvement,
        "interaction_patterns": stats,
        "ab_test_simulation": ab_result.__dict__,
        "note": (
            "SMOKE TEST on the tiny bundled fixture (8 users) — these numbers are not "
            "meaningful and exist only to validate the pipeline runs. ab_test_simulation "
            "uses synthetic CTRs, not a real experiment. Real results are in docs/results.md "
            "(MovieLens, sampled protocol). Reproduce with scripts/evaluate_models.py."
        ),
    }
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
