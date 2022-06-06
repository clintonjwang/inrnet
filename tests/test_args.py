import pytest

from torch import nn
from inrnet import args as args_module
from inrnet import util

@pytest.fixture
def args():
    args = {
        'slurm output dir': '~/code/inrnet/results',
        'job_id': 'manual',
        'data loading': {},
        'optimizer': {
            'type': 'AdamW',
            'beta1': .5,
            'learning_rate': '1e-5',
            'weight decay': '1e-3',
        },
    }
    args_module.infer_missing_args(args)
    return args

@pytest.fixture
def model():
    return nn.Linear(1,1)

def test_conversion(args):
    assert isinstance(args['optimizer']['G_learning_rate'], float)
    assert isinstance(args['optimizer']['weight decay'], float)

def test_optimizer(model, args):
    optim = util.get_optimizer(model.parameters(), args)
    assert hasattr(optim, 'param_groups')
    assert optim.param_groups[0]['lr'] == 1e-5
    assert optim.param_groups[0]['weight_decay'] == 1e-3
    assert optim.param_groups[0]['betas'][0] == .5
