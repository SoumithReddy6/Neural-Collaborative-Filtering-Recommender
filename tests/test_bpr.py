import torch

from ncf_recommender.losses import bpr_loss


def test_bpr_loss_lower_when_positive_score_is_higher():
    good = bpr_loss(torch.tensor([3.0]), torch.tensor([1.0]))
    bad = bpr_loss(torch.tensor([1.0]), torch.tensor([3.0]))
    assert good < bad

