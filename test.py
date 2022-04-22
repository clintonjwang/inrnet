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
    model = torchvision.models.vgg11_bn(pretrained=True)
    args = job_mgmt.get_job_args("dep1")
    data_loader = dataloader.get_inr_dataloader(args["data loading"])
    for inr, _ in data_loader:
        inr = to_black_box(inr)
        break
    img_shape = h,w = 352//32, 1216//32
    with torch.cuda.amp.autocast():
        img = inr.cuda().produce_image(h,w)
    x = torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda()

    discrete_layer = model.eval().features[0]
    InrNet = inn.conversion.translate_discrete_layer(discrete_layer)
    y = discrete_layer(x)

    coords = util.meshgrid_coords(*x.shape[-2:])
    out_inr = InrNet(inr)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = out_inr.eval()(coords)
            out = util.realign_values(out, coords_gt=coords, inr=out_inr)
            out = out.reshape(h,w,3).permute(2,0,1).unsqueeze(0)

    if y.shape != out.shape:
        print('shape mismatch')
        pdb.set_trace()
    if not torch.allclose(y.half(), out.half()):
        pdb.set_trace()

test_equivalence()
print("success")
# args = job_mgmt.get_job_args("dep1")
# train_depth_model(args)
