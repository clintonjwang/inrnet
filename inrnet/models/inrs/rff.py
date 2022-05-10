import torch
nn=torch.nn
import numpy as np
import cv2

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)
        x = x @ self._B.to(x.device)
        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

def fit_rff():
    target = torch.tensor(get_image(), device='cuda').unsqueeze(0).permute(0, 3, 1, 2)
    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), -1)
    xy_grid = torch.tensor(xy_grid, device='cuda').unsqueeze(0).permute(0, 3, 1, 2).float().contiguous()
    model = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 3, kernel_size=1, padding=0),
                nn.Sigmoid(),
            ).cuda()

    x = GaussianFourierFeatureTransform(2, 128, 10)(xy_grid)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
    for epoch in range(400):
        optimizer.zero_grad()
        generated = model(x)
        loss = torch.nn.functional.l1_loss(target, generated)
        loss.backward()
        optimizer.step()
