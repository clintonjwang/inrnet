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


def train_classifier(args):
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






def analyze_interpolate_grid_to_qmc():
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break
    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        logit_fxn = model(img_inr).cuda()
        qmc_coords = logit_fxn.generate_sample_points(sample_size=np.prod(img_shape), method='qmc')
        logit_fxn.toggle_grid_mode(True)
        grid_coords = logit_fxn.generate_sample_points(dims=img_shape)
        grid_logits = logit_fxn.eval()(grid_coords).cpu()
        grid_mask = (front_conv.mask_tracker, back_conv.mask_tracker)

        C = cdist(grid_coords.cpu().numpy(), qmc_coords.cpu().numpy())
        _, assigment = linear_sum_assignment(C)
        qmc_coords = qmc_coords[assigment]

        intermediate_logits = []
        intermediate_masks = []
        for alpha in np.arange(.05,1,.05):
            logit_fxn = model(img_inr).cuda()
            coords = alpha*qmc_coords + (1-alpha)*grid_coords
            intermediate_logits.append(logit_fxn.eval()(coords).cpu())
            intermediate_masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

        logit_fxn = model(img_inr).cuda()
        qmc_logits = logit_fxn.eval()(qmc_coords).cpu()
        qmc_mask = (front_conv.mask_tracker, back_conv.mask_tracker)

        torch.save((base_logits, grid_logits, grid_mask, qmc_logits, qmc_mask, intermediate_logits, intermediate_masks),
            osp.expanduser('~/code/diffcoord/temp/analyze_logit_mismatch.pt'))






def analyze_output_variance_rqmc():
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break
    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        logits = []
        masks = []
        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        for _ in range(20):
            logit_fxn = model(img_inr).cuda()
            coords = logit_fxn.generate_sample_points(sample_size=np.prod(img_shape), method='rqmc')
            logits.append(logit_fxn.eval()(coords).cpu())
            masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

        torch.save((base_logits, logits, masks), osp.expanduser('~/code/diffcoord/temp/output_variance_rqmc.pt'))


def analyze_change_resolution_grid_vs_qmc():
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break

    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        RES = np.round(np.logspace(4, 8, num=9, base=2)).astype(int)
        qmc_logits, grid_logits = [],[]
        qmc_masks, grid_masks = [],[]
        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        for res in RES:
            img_shape = (res,res)
            # logit_fxn = model(img_inr).cuda()
            # coords = logit_fxn.generate_sample_points(sample_size=np.prod(img_shape), method='qmc')
            # qmc_logits.append(logit_fxn.eval()(coords).cpu())
            # qmc_masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

            logit_fxn = model(img_inr).cuda()
            grid_coords = logit_fxn.generate_sample_points(dims=img_shape, method='grid')
            grid_logits.append(logit_fxn.eval()(grid_coords).cpu())
            grid_masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

            # torch.save((base_logits, grid_logits, grid_masks),
            #     osp.expanduser('~/code/diffcoord/temp/change_resolution_grid_only.pt'))
            torch.save((base_logits, grid_logits, grid_masks, qmc_logits, qmc_masks),
                osp.expanduser('~/code/diffcoord/temp/change_resolution_grid_vs_qmc.pt'))
            gc.collect()
            torch.cuda.empty_cache()


def analyze_divergence_over_depth():
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break
    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        logits = []
        masks = []
        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        for _ in range(20):
            logit_fxn = model(img_inr).cuda()
            coords = logit_fxn.generate_sample_points(sample_size=np.prod(img_shape), method='rqmc')
            logits.append(logit_fxn.eval()(coords).cpu())
            masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

        torch.save((base_logits, logits, masks),
            osp.expanduser('~/code/diffcoord/temp/analyze_logit_mismatch.pt'))
