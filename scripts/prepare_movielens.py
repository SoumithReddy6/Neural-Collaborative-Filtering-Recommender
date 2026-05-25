#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/interactions.csv")
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open("r", newline="") as source, output.open("w", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            target,
            fieldnames=["user_id", "item_id", "rating", "timestamp", "clicked", "session_id", "position"],
        )
        writer.writeheader()
        for row in reader:
            rating = float(row["rating"])
            writer.writerow(
                {
                    "user_id": f"u{row['userId']}",
                    "item_id": f"m{row['movieId']}",
                    "rating": rating,
                    "timestamp": row["timestamp"],
                    "clicked": 1 if rating >= args.positive_threshold else 0,
                    "session_id": f"u{row['userId']}-{int(row['timestamp']) // 3600}",
                    "position": 1,
                }
            )
    print(f"Wrote normalized interactions to {output}")


if __name__ == "__main__":
    main()
