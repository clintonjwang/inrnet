import torch, pdb, torchvision
nn=torch.nn
F=nn.functional
from time import time
import numpy as np

from inrnet.data import dataloader
from inrnet import jobs as job_mgmt
from inrnet import inn, util, experiments, models
from inrnet.models.inrs.siren import to_black_box

def test_equivalence():
    # D, D_B, G_A2B, G_B2A = models.cyclegan.load_pretrained_models('horse2zebra')
    # C = 3
    img_shape = h,w = 128,128
    # zz = torch.zeros(h*w, C)
    # zz[0,0] = 1
    # zz[-2,1] = 1
    # class dummy_inr(nn.Module):
    #     def forward(self, coords):
    #         return zz.to(dtype=coords.dtype, device=coords.device)
    # inr = inn.BlackBoxINR(evaluator=dummy_inr(), channels=C, input_dims=2, domain=(-1,1)).cuda()
    args = job_mgmt.get_job_args("dep1")
    data_loader = dataloader.get_inr_dataloader(args["data loading"])
    for inr, _ in data_loader:
        inr = to_black_box(inr).cuda()
        break

    with torch.no_grad():
        model = torchvision.models.efficientnet_b0(pretrained=True)
        # model = torchvision.models.squeezenet1_1(pretrained=True)
        # model.features[0].padding=(1,1)
        # model.features[5].padding=(1,1)
        # model.features[8].padding=(1,1)
        discrete_model = model.eval().cuda()
        # discrete_model = nn.Sequential(model.features[:5]).eval().cuda()
        # cv = nn.Conv2d(C,C,3,1,1,bias=False)
        # cv.weight.data.zero_()
        # cv.weight.data[:,:,1,1].fill_(1.)
        # cv = ResTest(C=C)
        # cv.block[0].weight.data.zero_()
        # cv.block[0].weight.data[:,:,1,1].fill_(1.)
        # discrete_model = nn.Sequential(cv).cuda().eval()
        # discrete_model = nn.Sequential(model.features[2][1]).cuda().eval()
        InrNet, output_shape = inn.conversion.translate_discrete_model(discrete_model, (h,w))

        coords = util.meshgrid_coords(h,w)
        t = time()
        out_inr = InrNet(inr)
        out_inr.toggle_grid_mode()
        out = out_inr.eval()(coords)
        print(f"produced INR in {np.round(time()-t, decimals=3)}s")

        if output_shape is not None:
            split = w // output_shape[-1]
            coords = util.first_split_meshgrid(h,w, split=split)
            out = util.realign_values(out, coords_gt=coords, inr=out_inr)
            out = out.reshape(*output_shape,-1).permute(2,0,1).unsqueeze(0)
        
        torch.cuda.empty_cache()
        img = inr.produce_image(h,w)#, split=2)
        x = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()
        y = discrete_model(x)

    if y.shape != out.shape:
        print('shape mismatch')
        pdb.set_trace()
    if not torch.allclose(y, out, rtol=.2, atol=.2):
        print('value mismatch:', np.nanmax((2*(y-out)/(out+y)).abs().cpu().numpy()), (y-out).abs().max().item())
        print((y-out).abs().mean().item(), y.abs().mean().item(), out.abs().mean().item())
        pdb.set_trace()

def test_equivalence_dummy():
    C = 1
    img_shape = h,w = 4,4
    zz = torch.zeros(h*w, C)
    zz[0,0] = 1
    zz[-6,0] = 1
    zz[-5,0] = 2
    zz[5,0] = 2
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return zz.to(dtype=coords.dtype, device=coords.device)
    inrs = inn.BlackBoxINR([dummy_inr()], channels=C, input_dims=2, domain=(-1,1)).cuda()
    with torch.no_grad():
        x = inrs.produce_images(h,w)
        conv = nn.Conv2d(1,1,3,1,padding=1,bias=False)
        conv.weight.data.fill_(0.)
        conv.weight.data[0,0,1,1].fill_(1.)
        # conv.bias.data.fill_(1.)

        discrete_layer = nn.Sequential(conv).cuda()
        # discrete_layer = nn.Sequential(nn.Upsample(scale_factor=2), nn.MaxPool2d(2,stride=2)).cuda()
        InrNet, output_shape = inn.conversion.translate_discrete_model(discrete_layer, (h,w))
        y = discrete_layer(x)

        coords = util.meshgrid_coords(h,w)
        out_inr = InrNet.cuda()(inrs)
        out_inr.toggle_grid_mode()
        out = out_inr.eval()(coords)
        
        if output_shape is not None:
            split = w // output_shape[-1]
            coords = util.first_split_meshgrid(h,w, split=split)
            out = util.realign_values(out, coords_gt=coords, inr=out_inr)
            out = out.reshape(*output_shape,-1).permute(2,0,1).unsqueeze(0)
        
    if y.shape != out.shape:
        print('shape mismatch')
        pdb.set_trace()
    if not torch.allclose(y, out, rtol=1e-5, atol=1e-3):
        print('value mismatch:', np.nanmax((2*(y-out)/(out+y)).abs().cpu().numpy()), (y-out).abs().max())
        pdb.set_trace()
    pdb.set_trace()

test_equivalence_dummy()
print("success")



def test_backprop():
    args = job_mgmt.get_job_args("dep1")
    torch.autograd.set_detect_anomaly(True)
    model = torchvision.models.efficientnet_b0(pretrained=True)
    C = 2
    img_shape = h,w = 4,4
    zz = torch.zeros(h*w, C)
    zz[0,0] = 1
    zz[-2,1] = 1
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return zz.to(dtype=coords.dtype, device=coords.device)
    inr = inn.BlackBoxINR(evaluator=dummy_inr(), channels=C, input_dims=2, domain=(-1,1)).cuda()

    # discrete_model = model.train().cuda()
    # InrNet, output_shape = inn.conversion.translate_discrete_model(discrete_model, (h,w))
    InrNet = inn.ChannelMixer(2,1, bias=False).cuda()
    InrNet.weight.data.zero_()
    optim = torch.optim.Adam(InrNet.parameters(), lr=.01)

    coords = util.meshgrid_coords(h,w)
    for _ in range(5):
        out_inr = InrNet(inr)
        out = out_inr(coords)
        optim.zero_grad()
        out.mean().backward()
        optim.step()
    pdb.set_trace()
        
# args = job_mgmt.get_job_args("dep1")
# train_depth_model(args)

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
