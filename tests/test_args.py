import pytest

from torch import nn
from inrnet.utils import args as args_module
from inrnet.utils import util

@pytest.fixture
def nn_model():
    return nn.Linear(1,1)

def test_conversion(args):
    assert isinstance(args['optimizer']['learning rate'], float)
    assert isinstance(args['optimizer']['weight decay'], float)

def test_optimizer(nn_model, args):
    optim = util.get_optimizer(nn_model, args)
    assert hasattr(optim, 'param_groups')
    assert optim.param_groups[0]['lr'] == 1e-5
    assert optim.param_groups[0]['weight_decay'] == 1e-3
    assert optim.param_groups[0]['betas'][0] == .5
