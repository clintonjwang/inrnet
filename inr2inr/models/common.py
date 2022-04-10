import torch
nn=torch.nn
F=nn.functional

def positional_encoding(coords, num_freqs=4):
    frequency_bands = 2.0 ** torch.linspace(
        0.0, num_freqs - 1, num_freqs,
        dtype=coords.dtype,
        device=coords.device,
    )
    encoding = []
    for freq in frequency_bands:
        encoding += [torch.sin(coords * freq), torch.cos(coords * freq)]
    return torch.cat(encoding, dim=-1)



class FC_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_channels,out_channels),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)

# from composer import functional as cf

# def modify_model(modifications, models):
#     for m in models:
#         if modifications["blurpool"]:
#             cf.apply_blurpool(m)
#         if modifications["squeeze-excite"]:
#             cf.apply_squeeze_excite(m)
#         if modifications["channels last"]:
#             cf.apply_channels_last(m)
#         if modifications["factorize layers"]:
#             cf.apply_factorization(m)
