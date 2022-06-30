import pytest
from torch import nn
from inrnet.utils import losses

@pytest.fixture
def loss_settings():
    return {'adversarial loss type':'WGAN'}

def test_losses(loss_settings):
    assert losses.CrossEntropy() is not None
    assert losses.L1_dist_inr() is not None
    assert losses.adv_loss_fxns(loss_settings) is not None