#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ncf_recommender.faiss_index import ItemVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default="artifacts/item_vectors.npz")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    artifact = np.load(args.vectors, allow_pickle=True)
    vectors = artifact["item_vectors"]
    index = ItemVectorIndex(vectors)
    result = index.search(vectors[0], args.top_k)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
