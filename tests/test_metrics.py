from ncf_recommender.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k


def test_ndcg_rewards_correct_ranking():
    assert ndcg_at_k([3, 2, 1], {3}, 3) == 1.0
    assert ndcg_at_k([1, 2, 3], {3}, 3) < 1.0


def test_hit_and_precision_at_k():
    ranked = [10, 11, 12]
    relevant = {11}
    assert hit_rate_at_k(ranked, relevant, 2) == 1.0
    assert precision_at_k(ranked, relevant, 2) == 0.5

