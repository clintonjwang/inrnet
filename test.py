import torch, pdb, torchvision
nn=torch.nn
F=nn.functional

from inrnet.data import dataloader
from inrnet import jobs as job_mgmt
from inrnet import inn, util
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

def test_equivalence():
    from inrnet.models.inrs.siren import to_black_box
    model = torchvision.models.vgg11(pretrained=True)
    args = job_mgmt.get_job_args("dep1")
    data_loader = dataloader.get_inr_dataloader(args["data loading"])
    for inr, _ in data_loader:
        inr = to_black_box(inr).cuda()
        break
    img_shape = h,w = 352//88, 1216//304
    with torch.cuda.amp.autocast():
        img = inr.produce_image(h,w)
    x = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()

    discrete_model = model.features[:2].cuda().eval()
    InrNet = nn.Sequential(
        *[inn.conversion.translate_discrete_layer(layer, img_shape).cuda() for layer in discrete_model]
    )
    y = discrete_model(x)

    coords = util.meshgrid_coords(h,w)
    out_inr = InrNet(inr)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = out_inr.eval()(coords)
            #coords = util.meshgrid_split_coords(h,w)[0]
            out = util.realign_values(out, coords_gt=coords, inr=out_inr)
            out = out.reshape(h,w,64).permute(2,0,1).unsqueeze(0)
            #out = out.reshape(h//2,w//2,3).permute(2,0,1).unsqueeze(0)

    if y.shape != out.shape:
        print('shape mismatch')
        pdb.set_trace()
    if not torch.allclose(y.half(), out.half(), rtol=.06, atol=.02):
        print('value mismatch:', ((y-out)/out).abs().max(), (y-out).abs().max())
        pdb.set_trace()

def test_equivalence_dummy():
    from inrnet.models.inrs.siren import to_black_box
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return (coords[:,0] > 0.1).to(dtype=coords.dtype).unsqueeze(-1)
    inr = inn.BlackBoxINR(evaluator=dummy_inr(), channels=3, input_dims=2, domain=(-1,1))
    img_shape = h,w = 4,4
    with torch.cuda.amp.autocast():
        img = inr.produce_image(h,w)
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0).cuda()
    conv = nn.Conv2d(1,1,(3,3),(1,1), padding=(1,1))
    # conv.weight.data.fill_(1.)
    # conv.weight.data[0,0,1,1].fill_(1.)
    # conv.bias.data.fill_(1.)

    discrete_layer = conv.cuda()
    InrNet = inn.conversion.translate_discrete_layer(discrete_layer, img_shape, smoothing=0.).cuda()
    y = discrete_layer(x)

    coords = util.meshgrid_coords(h,w)
    out_inr = InrNet(inr)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = out_inr.eval()(coords)
            out = util.realign_values(out, coords_gt=coords, inr=out_inr)
            out = out.reshape(h,w).unsqueeze(0).unsqueeze(0)

    if y.shape != out.shape:
        print('shape mismatch')
        pdb.set_trace()
    if not torch.allclose(y.half(), out.half(), rtol=1e-5, atol=1e-3):
        print('value mismatch')
        pdb.set_trace()

test_equivalence()
print("success")
# args = job_mgmt.get_job_args("dep1")
# train_depth_model(args)
