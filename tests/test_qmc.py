import pytest
import torch
from inrnet.inn import qmc

def test_qmc2d(qmc_2d_sequence):
    assert (qmc_2d_sequence.max() - 1).abs() < .1
    assert (qmc_2d_sequence.min() + 1).abs() < .1
    assert qmc_2d_sequence.shape == (128,2)