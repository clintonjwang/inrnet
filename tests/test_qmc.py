
from inrnet.inn import point_set, qmc

def test_qmc2d():
	coords = qmc.generate_quasirandom_sequence(d=2, n=128, bbox=(-1,1,-1,1), dtype=torch.float, device="cpu")
	assert (coords.max() - 1).abs() < 1e-3
	assert (coords.min() + 1).abs() < 1e-3