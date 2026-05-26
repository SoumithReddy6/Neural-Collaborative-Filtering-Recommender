#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncf_recommender.data import encode_interactions, leave_one_out_split, load_interactions, sample_bpr_triples
from ncf_recommender.training import build_model, evaluate_ranking, evaluate_ranking_sampled, train_bpr


def train_and_eval(
    model_type: str,
    interactions_path: str,
    epochs: int,
    protocol: str,
    num_negatives: int,
    negatives_per_positive: int,
) -> dict[str, float]:
    encoded = encode_interactions(load_interactions(interactions_path))
    train_pairs, test_pairs = leave_one_out_split(encoded)
    num_items = len(encoded.item_to_idx)
    triples = sample_bpr_triples(train_pairs, num_items, negatives_per_positive=negatives_per_positive)
    model = build_model(model_type, len(encoded.user_to_idx), num_items, embedding_dim=32)
    trained = train_bpr(model, triples, epochs=epochs)
    if protocol == "sampled":
        return evaluate_ranking_sampled(trained.model, test_pairs, train_pairs, num_items, k=10, num_negatives=num_negatives)
    return evaluate_ranking(trained.model, test_pairs, train_pairs, num_items, k=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default="data/samples/interactions.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--protocol", choices=["full", "sampled"], default="sampled",
                        help="sampled = standard NCF 1-vs-N protocol; full = rank against entire catalog.")
    parser.add_argument("--num-negatives", type=int, default=99, help="Negatives per positive at eval (sampled protocol).")
    parser.add_argument("--negatives-per-positive", type=int, default=4, help="Negatives per positive during BPR training.")
    parser.add_argument("--output", default="artifacts/model_comparison.json")
    args = parser.parse_args()

    results = {
        "ncf": train_and_eval("ncf", args.interactions, args.epochs, args.protocol, args.num_negatives, args.negatives_per_positive),
        "matrix_factorization": train_and_eval("mf", args.interactions, args.epochs, args.protocol, args.num_negatives, args.negatives_per_positive),
        "eval_protocol": args.protocol,
    }
    Path(args.output).parent.mkdir(exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
