import pytest
import torch

def test_qmc2d(qmc_2d_sequence256):
    assert (qmc_2d_sequence256.max() - 1).abs() < .1
    assert (qmc_2d_sequence256.min() + 1).abs() < .1
    assert qmc_2d_sequence256.shape == (256,2)