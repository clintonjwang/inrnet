from conftest import requirescuda
from inrnet.inn import point_set
# requirescuda

import pytest, torch, pdb, torchvision
nn=torch.nn
F=nn.functional
from time import time
import numpy as np

from inrnet import jobs as job_mgmt
from inrnet import inn, util
from inrnet.experiments import classify

import inrnet.inn.nets.convnext
import inrnet.models.convnext

@requirescuda
def test_equivalence():
    C = 32
    img_shape = h,w = 8,8
    zz = torch.randn(h*w, C)
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return zz.to(dtype=coords.dtype, device=coords.device)
    inr = inn.BlackBoxINR([dummy_inr()], channels=C, input_dims=2, domain=(-1,1)).cuda()
    # loader = get_inr_loader_for_cityscapes(1, 'train', img_shape)
    # for inr,_ in loader:
    #     break
    # inrnet.models.convnext.mini_convnext()
    # inrnet.inn.nets.convnext.translate_convnext_model((128,128))

    with torch.no_grad():
        # model = torchvision.models.efficientnet_b0(pretrained=True)
        model = classify.load_model_from_job('inet_nn_train')
        discrete_model = model[0][1][0].block[1:2].eval().cuda()
        InrNet, output_shape = inn.conversion.translate_discrete_model(discrete_model, (h,w))

        t = time()
        out_inr = InrNet(inr)
        out_inr.toggle_grid_mode()
        coords = out_inr.generate_sample_points(dims=(h,w))
        # coords = out_inr.generate_sample_points(sample_size=h*w)
        out = out_inr.eval()(coords)
        print(f"produced INR in {np.round(time()-t, decimals=3)}s")

        if output_shape is not None:
            out = out.reshape(*output_shape,-1).permute(2,0,1).unsqueeze(0)
        
        torch.cuda.empty_cache()
        x = inr.produce_images(h,w)
        y = discrete_model(x)

    assert y.shape != out.shape
    # print('shape mismatch')
    # pdb.set_trace()
    assert torch.allclose(y, out, rtol=.2, atol=.2)
    # print('value mismatch:', np.nanmax((2*(y-out)/(out+y)).abs().cpu().numpy()), (y-out).abs().max().item())
    # print((y-out).abs().mean().item(), y.abs().mean().item(), out.abs().mean().item())
    # pdb.set_trace()


@requirescuda
def test_equivalence_dummy():
    C = 1
    img_shape = h,w = 16,16
    zz = torch.zeros(h*w, C)
    zz[0,:] = 1
    zz[-4,:] = 1
    zz[-2,:] = 2
    zz[2,:] = 2
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return zz.to(dtype=coords.dtype, device=coords.device)
    inr = inn.BlackBoxINR([dummy_inr()], channels=C, input_dims=2, domain=(-1,1)).cuda()
    with torch.no_grad():
        x = inr.produce_images(h,w)
        conv = nn.Conv2d(1,2,3,1,padding=1,bias=False)
        # conv.weight.data.fill_(0.)
        # conv.weight.data[:,:,1,1].fill_(1.)
        norm = nn.BatchNorm2d(2)

        discrete_model = nn.Sequential(conv, norm, nn.LeakyReLU(inplace=True)).cuda().eval()
        # discrete_model = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.MaxPool2d(2,stride=2)).cuda()
        InrNet, output_shape = inn.conversion.translate_discrete_model(discrete_model, (h,w))
        y = discrete_model(x)

        coords = point_set.meshgrid_coords(h,w)
        out_inr = InrNet.cuda()(inr)
        out_inr.toggle_grid_mode(True)
        out = out_inr.eval()(coords)
        
        if output_shape is not None:
            out = util.realign_values(out, inr=out_inr)
            out = out.reshape(1,*output_shape,-1).permute(0,3,1,2)
        
    if y.shape != out.shape:
        print('shape mismatch')
        pdb.set_trace()
    if not torch.allclose(y, out, rtol=1e-5, atol=1e-3):
        print('value mismatch:', np.nanmax((2*(y-out)/(out+y)).abs().cpu().numpy()), (y-out).abs().max())
        pdb.set_trace()
    pdb.set_trace()


@requirescuda
def test_backprop():
    job_mgmt.get_job_args("dep1")
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

    coords = point_set.meshgrid_coords(h,w)
    for _ in range(5):
        out_inr = InrNet(inr)
        out = out_inr(coords)
        optim.zero_grad()
        out.mean().backward()
        optim.step()
    pdb.set_trace()
        
# args = job_mgmt.get_job_args("dep1")
# train_depth_model(args)

@requirescuda
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

@requirescuda
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
