import os, pdb, torch, gc
osp = os.path
nn = torch.nn
F = nn.functional
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models

from inrnet.data import dataloader, inet
from inrnet import inn, util, losses, jobs as job_mgmt

def load_pretrained_model(args):
    net_args = args["network"]
    pretrained = net_args['pretrained']
    img_shape = args["data loading"]["image shape"]
    if isinstance(pretrained, str):
        base = load_model_from_job(pretrained)
    else:
        # if net_args["type"] == "inr-effnet-b0":
        #     base = torchvision.models.efficientnet_b0(pretrained=pretrained)
        # elif net_args["type"] == "effnet-b0":
        #     return torchvision.models.efficientnet_b0(pretrained=pretrained)
        out = nn.Linear(24, args["data loading"]['classes'])
        nn.init.kaiming_uniform_(out.weight, mode='fan_in')
        out.bias.data.zero_()
        if net_args["type"] == "effnet-s3":
            m = torchvision.models.efficientnet_b0(pretrained=pretrained)
            return nn.Sequential(m.features[:3],
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(1), out)

        elif net_args["type"] == "inr-effnet-s3":
            m = torchvision.models.efficientnet_b0(pretrained=pretrained)
            base = nn.Sequential(
                m.features[:3],
                nn.AdaptiveAvgPool2d(output_size=1), out)

        elif net_args["type"] == "inr-effnet-mlp":
            m = torchvision.models.efficientnet_b0(pretrained=pretrained)
            base = nn.Sequential(
                m.features[:3],
                nn.AdaptiveAvgPool2d(output_size=1), out)
            InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
            inn.inrnet.replace_conv_kernels(InrNet, k_type='mlp',
                k_ratio=net_args["kernel expansion ratio"])

        else:
            raise NotImplementedError

    InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
    if net_args['frozen'] is True:
        inn.inrnet.freeze_layer_types(InrNet)
    return InrNet

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/best.pth")
    model = load_pretrained_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def rand_inr_generator_circle(bsz, order=16):
    def circle_forward(coords, coeffs, bias):
        # coords - (B,N,1)
        # coeffs - (B,2,O)
        # bias - (B,1)
        nx = 2*torch.pi * torch.arange(1,order, device='cuda') * coords #(B,N,O)
        return (coeffs[:,0:1]*torch.cos(nx) + coeffs[:,1:2]*torch.sin(nx)).sum(-1) + bias
    while True:
        fourier_coeffs = torch.randn(bsz, 2, order, device='cuda') / torch.arange(1,order, device='cuda').pow(2).unsqueeze(-1)
        bias = torch.randn(bsz, 1, device='cuda')/2
        inrs = inn.BlackBoxINR(evaluator=partial(circle_forward, coeffs=fourier_coeffs, bias=bias))
        yield inrs

# def rand_inr_generator_2d(bsz):
#     while True:

#     return

def basic_mlp_net():
    inn.MLPConv(3,16,(.2,.2))
    inn.MaxPool(2)
    inn.MLPConv(16,32,(.2,.2))
    inn.MLPConv(32,64,(.2,.2))
    inn.Upsample(4)

def train_fitting(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())
    top5_tracker = util.MetricTracker("top5", function=top5)
    top1_tracker = util.MetricTracker("top1", function=top1)
    bsz = dl_args['batch size']

    model = basic_mlp_net().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])
    for img_inr, segs in rand_inr_generator_2d(bsz):
        global_step += 1
        N = dl_args["sample points"]
        for img_inr, labels in data_loader:
            global_step += 1
            logit_fxn = model(img_inr)
            coords = logit_fxn.generate_sample_points(sample_size=N, method='rqmc')
            logits = logit_fxn(coords)
            backprop(model)
            if global_step >= args["optimizer"]["max steps"]:
                break


    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def test_inr_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    top3, top1 = 0,0
    origin = args['target_job']
    model = load_model_from_job(origin).cuda().eval()
    orig_args = job_mgmt.get_job_args(origin)

    if orig_args["network"]['type'].startswith('inr'):
        N = dl_args["sample points"]
        for img_inr, labels in data_loader:
            with torch.no_grad():
                logit_fxn = model(img_inr).cuda().eval()
                coords = logit_fxn.generate_sample_points(sample_size=N)
                logits = logit_fxn(coords)
                pred_cls = logits.topk(k=3).indices
                top3 += (labels.unsqueeze(1) == pred_cls).amax(1).float().sum().item()
                top1 += (labels == pred_cls[:,0]).float().sum().item()

    else:
        for img_inr, labels in data_loader:
            with torch.no_grad():
                img = img_inr.produce_images(*dl_args['image shape'])
                logits = model(img)
                pred_cls = logits.topk(k=3).indices
                top3 += (labels.unsqueeze(1) == pred_cls).amax(1).float().sum().item()
                top1 += (labels == pred_cls[:,0]).float().sum().item()

    torch.save((top1, top3), osp.join(paths["job output dir"], "stats.pt"))
