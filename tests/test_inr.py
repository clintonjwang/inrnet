import pytest, torch
from inrnet import inn

def test_inr_parent(inr16x16, inr_classifier): #qmc_2d_sequence256
    if not torch.cuda.is_available():
        pytest.skip('no cuda')
    inr_cls = inr_classifier(inr16x16)
    inr_cls.change_sample_mode('grid')
    assert next(inr_cls.evaluator_iter()).sample_mode == 'grid'
    assert inr_cls.parent(n=5).sample_mode == 'grid'
