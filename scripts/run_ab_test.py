#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncf_recommender.ab_testing import simulate_ab_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ctr", type=float, default=0.122)
    parser.add_argument("--treatment-ctr", type=float, default=0.146)
    parser.add_argument("--users-per-arm", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/ab_test_results.json")
    args = parser.parse_args()

    result = simulate_ab_test(args.baseline_ctr, args.treatment_ctr, args.users_per_arm)
    Path(args.output).parent.mkdir(exist_ok=True)
    Path(args.output).write_text(json.dumps(result.__dict__, indent=2) + "\n")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
