from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class ABTestResult:
    baseline_ctr: float
    treatment_ctr: float
    relative_lift: float
    z_score: float
    p_value: float
    significant: bool


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    p_a = success_a / n_a
    p_b = success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if standard_error == 0:
        return 0.0, 1.0
    z_score = (p_b - p_a) / standard_error
    p_value = 2 * (1 - _normal_cdf(abs(z_score)))
    return z_score, p_value


def simulate_ab_test(
    baseline_ctr: float,
    treatment_ctr: float,
    users_per_arm: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> ABTestResult:
    rng = random.Random(seed)
    baseline_successes = sum(1 for _ in range(users_per_arm) if rng.random() < baseline_ctr)
    treatment_successes = sum(1 for _ in range(users_per_arm) if rng.random() < treatment_ctr)
    observed_baseline = baseline_successes / users_per_arm
    observed_treatment = treatment_successes / users_per_arm
    z_score, p_value = two_proportion_z_test(baseline_successes, users_per_arm, treatment_successes, users_per_arm)
    relative_lift = (observed_treatment - observed_baseline) / observed_baseline if observed_baseline else 0.0
    return ABTestResult(observed_baseline, observed_treatment, relative_lift, z_score, p_value, p_value < alpha)

