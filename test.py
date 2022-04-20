import torch, pdb
nn=torch.nn
F=nn.functional

from inrnet import jobs as job_mgmt
from inrnet import inn
from inrnet.experiments.depth import train_depth_model

def test_layers():
    siren = nn.Linear(2,3).cuda()
    inr = inn.BlackBoxINR(siren, channels=3, input_dims=2).cuda()
    model = nn.Sequential(
       inn.Conv(3,8, radius=.4, input_dims=2, dropout=.2),
       inn.ChannelNorm(8),
       inn.ReLU(),
       inn.AdaptiveChannelMixer(8,4, input_dims=2),
       inn.AvgPool(radius=.4),
    ).cuda().train()
    new_inr = model(inr)
    with torch.no_grad():
        shape = new_inr(torch.randn(64,2).cuda()).shape
        assert shape == (64,4), f"shape is {shape}"
    new_inr(torch.randn(64,2).cuda()).sum().backward()

def test_network():
    siren = nn.Linear(2,3).cuda()
    inr = inn.BlackBoxINR(siren, channels=3, input_dims=2).cuda()
    model = inn.nets.Conv4(3,4).cuda().train()
    new_inr = model(inr)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            shape = new_inr(torch.randn(8,2).cuda()).shape
            assert shape == (1,4), f"shape is {shape}"
        new_inr(torch.randn(8,2).cuda()).sum().backward()

test_network()
print("success")
# args = job_mgmt.get_job_args("dep1")
# train_depth_model(args)
