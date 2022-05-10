import os, pdb, torch
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from inrnet.data import dataloader
from inrnet import inn, util, losses, jobs as job_mgmt
import inrnet.inn.nets.convnext
import inrnet.models.convnext

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_float = mtr.ScaleIntensity()

def load_pretrained_model(args):
    net_args = args["network"]
    pretrained = net_args['pretrained']
    if isinstance(pretrained, str):
        raise NotImplementedError
        base = load_model_from_job(pretrained)
    else:
        if net_args["type"] == "convnext":
            return inrnet.models.convnext.mini_convnext()
        elif net_args["type"] == "inr-convnext":
            if pretrained is False:
                print('from scratch not implemented yet')
            InrNet = inrnet.inn.nets.convnext.translate_convnext_model(args["data loading"]["image shape"])
        elif net_args["type"] == "inr-mlpconv":
            InrNet = inrnet.inn.nets.convnext.translate_convnext_model(args["data loading"]["image shape"])
            inn.inrnet.replace_conv_kernels(InrNet, k_type='mlp')
        else:
            raise NotImplementedError
            
    if net_args['frozen'] is True:
        inn.inrnet.freeze_layer_types(InrNet)
    return InrNet

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/best.pth")
    model = load_pretrained_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def mean_iou(pred_seg, gt_seg):
    # pred_seg [B*N], gt_seg [B*N,C]
    iou_per_channel = (pred_seg & gt_seg).sum(0) / (pred_seg | gt_seg).sum(0)
    return iou_per_channel.mean()

def pixel_acc(pred_seg, gt_seg):
    return (pred_seg & gt_seg).sum() / pred_seg.size(0)

def train_segmenter(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())#nn.BCEWithLogitsLoss())
    iou_tracker = util.MetricTracker("mean IoU", function=mean_iou)
    acc_tracker = util.MetricTracker("pixel accuracy", function=pixel_acc)
    bsz = dl_args['batch size']

    def backprop(network):
        # loss = loss_tracker(logits, segs.float()) #BCE
        maxes, gt_labels = segs.max(-1)
        loss = loss_tracker(logits[maxes != 0], gt_labels[maxes != 0]) #cross entropy
        if torch.isnan(loss):
            raise ValueError('nan loss')
        pred_seg = logits.max(-1).indices
        pred_1hot = F.one_hot(pred_seg, num_classes=segs.size(-1)).bool()
        iou = iou_tracker(pred_1hot[maxes != 0], segs[maxes != 0]).item()
        acc = acc_tracker(pred_1hot[maxes != 0], segs[maxes != 0]).item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if global_step % 20 == 0:
            print(np.round(loss.item(), decimals=3), "; iou:", np.round(iou, decimals=3),
                "; acc:", np.round(acc*100, decimals=2),
                flush=True)

        if global_step % 100 == 0:
            torch.save(network.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
            iou_tracker.plot_running_average(path=paths["job output dir"]+"/plots/iou.png")
            acc_tracker.plot_running_average(path=paths["job output dir"]+"/plots/acc.png")

            rgb = img_inr.produce_images(*dl_args['image shape'])[0]
            path = paths["job output dir"]+f"/imgs/{global_step}.png"
            save_example_segs(path, rgb, pred_seg[0].reshape(*rgb.shape[-2:]), gt_labels[0].reshape(*rgb.shape[-2:]))

    model = load_pretrained_model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])
    for img_inr, segs in data_loader:
        global_step += 1
        
        if args["network"]['type'].startswith('inr'):
            if global_step == args["optimizer"]["max steps"]//2:
                inrnet.inn.nets.convnext.enable_cn_blocks(model)

            seg_inr = model(img_inr)
            if dl_args['sample type'] == 'grid':
                segs = segs.flatten(start_dim=2).transpose(2,1)
                seg_inr.toggle_grid_mode(True)
                coords = seg_inr.generate_sample_points(dims=dl_args['image shape'])
            else:
                segs = get_seg_at_coords(coords)
                coords = seg_inr.generate_sample_points(sample_size=dl_args["sample points"], method=dl_args['sample type'])
            logits = seg_inr.cuda()(coords)
            if dl_args['sample type'] == 'grid':
                logits = util.realign_values(logits, coords=coords)

        else:
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img).flatten(start_dim=2).transpose(2,1)
            segs = segs.flatten(start_dim=2).transpose(2,1)

        backprop(model)
        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def test_inr_segmenter(args):
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

import imgviz
def save_example_segs(path, rgb, pred_seg, gt_seg, class_names=('ground', 'building', 'traffic', 'nature', 'sky', 'human', 'vehicle')):
    label_names = [
        "{}:{}".format(i, n) for i, n in enumerate(class_names)
    ]
    labelviz_pred = imgviz.label2rgb(pred_seg.cpu())#, label_names=label_names, font_size=6, loc="rb")
    labelviz_gt = imgviz.label2rgb(gt_seg.cpu())#, label_names=label_names, font_size=6, loc="rb")
    rgb = rescale_float(rgb.cpu().permute(1,2,0))

    plt.figure(dpi=400)

    plt.subplot(131)
    plt.title("rgb")
    plt.imshow(rgb)
    plt.axis("off")
    plt.subplot(132)
    plt.title("pred")
    plt.imshow(labelviz_pred)
    plt.axis("off")
    plt.subplot(133)
    plt.title("gt")
    plt.imshow(labelviz_gt)
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.imsave(path, img)
    plt.close()

def save_figure():
    with torch.no_grad():
        h,w = H//4, W//4
        tensors = [torch.linspace(-1, 1, steps=h), torch.linspace(-1, 1, steps=w)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij', device='cuda'), dim=-1)
        xy_grid = mgrid.reshape(-1, 2).half()
        InrNet.eval()
        z_pred = Seg_inr(xy_grid)
        z_pred = rescale_float(z_pred.reshape(h,w).cpu().float().numpy())
        plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_z.png"), z_pred, cmap="gray")

        del z_pred, Seg_inr
        torch.cuda.empty_cache()
        rgb = img_inr.evaluator(xy_grid)
        rgb = rescale_float(rgb.reshape(h,w, 3).cpu().float().numpy())
        plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_rgb.png"), rgb)

        torch.cuda.empty_cache()
        InrNet.train()
        

def get_seg_at_coords():
    return