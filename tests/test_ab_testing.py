from ncf_recommender.ab_testing import simulate_ab_test, two_proportion_z_test


def test_ab_test_reports_positive_lift():
    result = simulate_ab_test(0.10, 0.14, users_per_arm=1000, seed=7)
    assert result.relative_lift > 0


def test_two_proportion_z_test_returns_probability():
    _, p_value = two_proportion_z_test(100, 1000, 140, 1000)
    assert 0 <= p_value <= 1

