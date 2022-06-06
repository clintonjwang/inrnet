import pytest
import torch
from inrnet.inn import point_set, qmc

@pytest.fixture
def qmc_2d_sequence():
    coords = qmc.generate_quasirandom_sequence(d=2, n=128, bbox=(-1,1,-1,1), dtype=torch.float, device="cpu")
    return coords

def test_qmc2d(qmc_2d_sequence):
    assert (qmc_2d_sequence.max() - 1).abs() < .1
    assert (qmc_2d_sequence.min() + 1).abs() < .1
    assert qmc_2d_sequence.shape == (128,2)