#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncf_recommender.data import encode_interactions, leave_one_out_split, load_interactions, sample_bpr_triples
from ncf_recommender.training import build_model, evaluate_ranking, train_bpr


def train_and_eval(model_type: str, interactions_path: str, epochs: int) -> dict[str, float]:
    encoded = encode_interactions(load_interactions(interactions_path))
    train_pairs, test_pairs = leave_one_out_split(encoded)
    triples = sample_bpr_triples(train_pairs, len(encoded.item_to_idx))
    model = build_model(model_type, len(encoded.user_to_idx), len(encoded.item_to_idx), embedding_dim=32)
    trained = train_bpr(model, triples, epochs=epochs)
    return evaluate_ranking(trained.model, test_pairs, train_pairs, len(encoded.item_to_idx), k=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default="data/samples/interactions.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", default="artifacts/model_comparison.json")
    args = parser.parse_args()

    results = {
        "ncf": train_and_eval("ncf", args.interactions, args.epochs),
        "matrix_factorization": train_and_eval("mf", args.interactions, args.epochs),
    }
    Path(args.output).parent.mkdir(exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
