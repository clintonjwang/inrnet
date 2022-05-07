import os, pdb, torch, gc
osp = os.path
nn = torch.nn
F = nn.functional
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
            base = nn.Sequential(m.features[:3],
                nn.AdaptiveAvgPool2d(output_size=1), out)

        elif net_args["type"] == "inr-effnet-mlp":
            m = torchvision.models.efficientnet_b0(pretrained=pretrained)
            base = nn.Sequential(m.features[:3],
                nn.AdaptiveAvgPool2d(output_size=1), out)
            InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
            inn.conversion.replace_conv_kernels(InrNet, k_type='mlp')

        else:
            raise NotImplementedError
        
    InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
    return InrNet

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/best.pth")
    model = load_pretrained_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model


def finetune_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())
    top5 = lambda pred_cls, labels: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()
    top1 = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()
    top5_tracker = util.MetricTracker("top5", function=top5)
    top1_tracker = util.MetricTracker("top1", function=top1)
    bsz = dl_args['batch size']

    def backprop(network):
        loss = loss_tracker(logits, labels)
        pred_cls = logits.topk(k=5).indices
        top_5 = top5_tracker(pred_cls, labels).item()
        top_1 = top1_tracker(pred_cls, labels).item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if global_step % 20 == 0:
            print(np.round(loss.item(), decimals=3), "; top_5:", np.round(top_5, decimals=2),
                "; top_1:", np.round(top_1, decimals=2),
                flush=True)
        if global_step % 100 == 0:
            torch.save(network.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
            top1_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top1.png")
            top5_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top5.png")
        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

    model = load_pretrained_model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])
    if args["network"]['type'].startswith('inr'):
        N = dl_args["sample points"]
        for img_inr, labels in data_loader:
            global_step += 1
            logit_fxn = model(img_inr)
            coords = logit_fxn.generate_sample_points(sample_size=N, method='rqmc')
            logits = logit_fxn(coords)
            backprop(model)
            if global_step >= args["optimizer"]["max steps"]:
                break

    else:
        for img_inr, labels in data_loader:
            global_step += 1
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img)
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



def analyze_inr_error(dataset):
    img_shape = (256,256)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    base = load_model_from_job(pretrained)
    for img_inr, label in data_loader:
        break

def analyze_logit_mismatch():
    img_shape = (128,128)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        break
    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img)

        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        logit_fxn = model(img_inr).cuda()
        logit_fxn.toggle_grid_mode(True)
        coords = logit_fxn.generate_sample_points(dims=img_shape)
        grid_logits = logit_fxn.eval()(coords)
        pdb.set_trace()

        logit_fxn = model(img_inr).cuda()
        dx = 2/128 / 32 # 1/32 of a pixel
        coords = torch.clamp(coords + torch.randn_like(coords)*dx, min=-1, max=1)
        pert_logits1 = logit_fxn.eval()(coords)


        gc.collect()
        torch.cuda.empty_cache()
        logit_fxn = model(img_inr).cuda()
        coords = torch.clamp(coords + torch.randn_like(coords)*dx, min=-1, max=1)
        pert_logits2 = logit_fxn.eval()(coords)


        logit_fxn = model(img_inr).cuda()
        coords = logit_fxn.generate_sample_points(sample_size=np.prod(img_shape), method='qmc')
        test_logits = logit_fxn.eval()(coords)

        torch.save((base_logits, grid_logits, pert_logits1, pert_logits2, test_logits),
            osp.expanduser('~/code/diffcoord/temp/analyze_logit_mismatch.pt'))

def analyze_logit_drift():
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(12, 'test')
    for img_inr, labels in data_loader:
        break

    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img)

    model, _ = inn.conversion.translate_discrete_model(base, img_shape)
    logit_fxn = model(img_inr)
    coords = logit_fxn.generate_sample_points(sample_size=N, method='rqmc')
    logits = logit_fxn(coords)

    torch.save((top1, top3), osp.join(paths["job output dir"], "stats.pt"))
