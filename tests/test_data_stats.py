from ncf_recommender.data import encode_interactions, leave_one_out_split, load_interactions, sample_bpr_triples
from ncf_recommender.stats import interaction_summary


def test_data_pipeline_builds_training_triples():
    interactions = load_interactions("data/samples/interactions.csv")
    encoded = encode_interactions(interactions)
    train, test = leave_one_out_split(encoded)
    triples = sample_bpr_triples(train, len(encoded.item_to_idx), negatives_per_positive=1)
    assert train
    assert test
    assert triples


def test_interaction_summary_has_ctr_and_session_depth():
    summary = interaction_summary(load_interactions("data/samples/interactions.csv"))
    assert summary["ctr"] > 0
    assert summary["avg_session_depth"] >= 1

