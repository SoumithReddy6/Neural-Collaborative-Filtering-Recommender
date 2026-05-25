#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/experiment_manifest.csv")
    args = parser.parse_args()

    models = ["ncf", "mf"]
    learning_rates = [0.0005, 0.001, 0.003]
    embedding_dims = [32, 64, 128]
    negatives = [2, 4, 8]
    seeds = [13, 21, 42]

    rows = []
    for run_id, values in enumerate(product(models, learning_rates, embedding_dims, negatives, seeds), start=1):
        model, lr, dim, neg, seed = values
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "learning_rate": lr,
                "embedding_dim": dim,
                "negatives_per_positive": neg,
                "seed": seed,
                "tracking_backend": "mlflow",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiment configs to {output}")


if __name__ == "__main__":
    main()
