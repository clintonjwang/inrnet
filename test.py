import torch, pdb
nn=torch.nn
F=nn.functional

from inrnet import jobs as job_mgmt
from inrnet import inn
from inrnet.experiments.depth import train_depth_model

def test_1():
    siren = nn.Linear(2,3).cuda()
    inr = inn.BlackBoxINR(siren, channels=3, input_dims=2).cuda()
    model = nn.Sequential(
       inn.Conv(3,8, radius=.4, input_dims=2),
       inn.BatchNorm(8),
       inn.ReLU(),
       inn.ChannelMixing(8,4),
       inn.AvgPool(radius=.4),
    ).cuda().train()
    new_inr = model(inr)
    with torch.no_grad():
       assert new_inr(torch.randn(6,2).cuda()).shape == (6,4)
    new_inr(torch.randn(6,2).cuda()).sum().backward()
    
    inr = inn.BlackBoxINR(siren, channels=3, input_dims=2).cuda()
    new_inr = model(inr)
    new_inr(torch.randn(6,2).cuda()).sum().backward()
    print("success")

test_1()
# args = job_mgmt.get_job_args("depth")
# train_depth_model(args)
