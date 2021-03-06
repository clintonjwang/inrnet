import os, pdb, torch
import wandb

from inrnet.utils import losses, util
from inrnet.utils import jobs as job_mgmt
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from inrnet.utils import args as args_module
from inrnet.data import dataloader
from inrnet import inn
from inrnet.inn import point_set
import inrnet.inn.nets.convnext
import inrnet.models.convnext
import inrnet.models.common
import inn.nets.inr2inr

rescale_float = mtr.ScaleIntensity()
DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

def load_model(args):
    net_args = args["network"]
    if net_args["type"].startswith("inr"):
        sampler = point_set.get_sampler_from_args(args['data loading'])
    kwargs = dict(in_channels=3, out_channels=7)
    if net_args["type"] == "inr-3":
        return inn.nets.inr2inr.ISeg3(sampler=sampler, **kwargs)
    elif net_args["type"] == "inr-5":
        return inn.nets.inr2inr.ISeg5(sampler=sampler, **kwargs)
    elif net_args["type"] == "cnn-3":
        return inrnet.models.common.Seg3(**kwargs)
    elif net_args["type"] == "cnn-5":
        return inrnet.models.common.Seg5(**kwargs)

    pretrained = net_args['pretrained']
    if isinstance(pretrained, str):
        raise NotImplementedError
        load_model_from_job(pretrained)
    else:
        if net_args["type"] == "convnext":
            return inrnet.models.convnext.mini_convnext()
        elif net_args["type"] == "inr-convnext":
            InrNet = inrnet.inn.nets.convnext.translate_convnext_model(
                    args["data loading"]["image shape"], sampler=sampler)
        elif net_args["type"] == "inr-mlpconv":
            InrNet = inrnet.inn.nets.convnext.translate_convnext_model(
                    args["data loading"]["image shape"], sampler=sampler)
            inn.inrnet.replace_conv_kernels(InrNet, k_type='mlp')
        else:
            raise NotImplementedError
            
    return InrNet

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/inrnet/results/{origin}/weights/model.pth")
    model = load_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def get_seg_at_coords(seg, coords):
    coo = torch.floor(coords).long()
    return seg[...,coo[:,0], coo[:,1]].transpose(1,2)

def train_segmenter(args: dict) -> None:
    if not args['no_wandb']:
        wandb.init(project="inrnet", job_type="train", name=args["job_id"],
            config=wandb.helper.parse_config(args, exclude=['job_id']))
        args = args_module.get_wandb_config()
    paths = args["paths"]
    dl_args = args["data loading"]
    global_step = 0
    trainsegdist = torch.load(DS_DIR+'/inrnet/cityscapes/trainsegdist.pt')
    weight = trainsegdist.sum() / trainsegdist
    loss_fxn = nn.CrossEntropyLoss(weight=weight.cuda())
    if dl_args['sample type'] == 'masked':
        dl_args['batch size'] = 1 #cannot handle different masks per datapoint
    data_loader = dataloader.get_inr_dataloader(dl_args)

    model = load_model(args).cuda()
    optimizer = util.get_optimizer(model, args)
    for img_inr, segs in data_loader:
        global_step += 1
        
        if args["network"]['type'].startswith('inr'):
            logit_inr = model(img_inr)

            if dl_args['sample type'] == 'grid':
                seg_gt = segs.flatten(start_dim=2).transpose(2,1)
                logit_inr.sort()
            else:
                seg_gt = get_seg_at_coords(segs, logit_inr.coords)
            logits = logit_inr.values
                
        else:
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img).flatten(start_dim=2).transpose(2,1)
            seg_gt = segs.flatten(start_dim=2).transpose(2,1)

        # loss = loss_tracker(logits, seg_gt.float()) #BCE
        maxes, gt_labels = seg_gt.max(-1)
        loss = loss_fxn(logits[maxes != 0], gt_labels[maxes != 0]) #cross entropy
        if torch.isnan(loss):
            raise ValueError('nan loss')
        pred_seg = logits.max(-1).indices
        pred_1hot = F.one_hot(pred_seg, num_classes=seg_gt.size(-1)).bool()
        iou = losses.mean_iou(pred_1hot[maxes != 0], seg_gt[maxes != 0]).item()
        acc = losses.pixel_acc(pred_1hot[maxes != 0], seg_gt[maxes != 0]).item()
        wandb.log({"train_loss": loss.item(), "train_mIoU": iou, "train_PixAcc": acc})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if global_step % 20 == 0:
            print(np.round(loss.item(), decimals=3), "; iou:", np.round(iou, decimals=3),
                "; acc:", np.round(acc*100, decimals=2),
                flush=True)

        if global_step % 100 == 0:
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))

            path = paths["job output dir"]+f"/imgs/{global_step}.png"
            with torch.no_grad():
                rgb = img_inr.produce_images(*dl_args['image shape'])[0]
                gt_labels += 1
                gt_labels[maxes == 0] = 0
                pred_seg += 1
                pred_seg[maxes == 0] = 0
                save_example_segs(path, rgb, pred_seg[0].reshape(*rgb.shape[-2:]), gt_labels[0].reshape(*rgb.shape[-2:]))

        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))


def test_inr_segmenter(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    if dl_args['sample type'] == 'masked':
        dl_args['batch size'] = 1 #cannot handle different masks per datapoint
    data_loader = dataloader.get_inr_dataloader(dl_args)

    origin = args['target_job']
    model = load_model_from_job(origin).cuda().eval()
    orig_args = job_mgmt.get_job_args(origin)
    ix = 0
    for img_inr, segs in data_loader:
        ix += 1
        with torch.no_grad():
            if orig_args["network"]['type'].startswith('inr'):
                seg_inr = model(img_inr).eval()
                if dl_args['sample type'] == 'masked':
                    seg_inr.change_sample_mode('masked')
                    coords = point_set.generate_masked_sample_points(mask=(segs.amax(1) == True),
                        sample_size=dl_args["sample points"])
                    seg_gt = get_seg_at_coords(segs, coords)

                elif dl_args['sample type'] == 'grid':
                    seg_gt = segs.flatten(start_dim=2).transpose(2,1)
                    seg_inr.change_sample_mode('grid')
                    coords = seg_inr.generate_sample_points(dims=dl_args['image shape'])
                
                else:
                    coords = seg_inr.generate_sample_points(sample_size=dl_args["sample points"],
                        method=dl_args['sample type'])
                    seg_gt = get_seg_at_coords(segs, coords)
                    
                logits = seg_inr(coords)
                logits = util.realign_values(logits, coords=coords)

            else:
                seg_gt = segs.flatten(start_dim=2).transpose(2,1)
                img = img_inr.produce_images(*dl_args['image shape'])
                logits = model(img).flatten(start_dim=2).transpose(2,1)

            maxes, gt_labels = seg_gt.max(-1)
            pred_seg = logits.max(-1).indices
            pred_1hot = F.one_hot(pred_seg, num_classes=seg_gt.size(-1)).bool()
            iou = losses.mean_iou(pred_1hot[maxes != 0], seg_gt[maxes != 0]).item()
            acc = losses.pixel_acc(pred_1hot[maxes != 0], seg_gt[maxes != 0]).item()
            wandb.log({"test_mIoU": iou, "test_PixAcc": acc})

        if dl_args['sample type'] == 'grid':
            rgb = img_inr.produce_images(*dl_args['image shape'])
            gt_labels += 1
            pred_seg += 1
            gt_labels[maxes == 0] = 0
            pred_seg[maxes == 0] = 0
            for i in range(3):
                path = paths["job output dir"]+f"/imgs/{ix}_{i}.png"
                save_example_segs(path, rgb[i], pred_seg[i].reshape(*rgb.shape[-2:]), gt_labels[i].reshape(*rgb.shape[-2:]))

import imgviz
def save_example_segs(path, rgb, pred_seg, gt_seg, class_names=('ground', 'building', 'traffic', 'nature', 'sky', 'human', 'vehicle')):
    label_names = [
        "{}:{}".format(i, n) for i, n in enumerate(class_names)
    ]
    labelviz_pred = imgviz.label2rgb(pred_seg.cpu())#, label_names=label_names, font_size=6, loc="rb")
    labelviz_gt = imgviz.label2rgb(gt_seg.cpu())#, label_names=label_names, font_size=6, loc="rb")
    rgb = rescale_float(rgb.cpu().permute(1,2,0))

    # kwargs = dict(bbox_inches='tight', transparent="True", pad_inches=0)
    plt.figure(dpi=400)
    plt.tight_layout()
    plt.subplot(131)
    # plt.title("rgb")
    plt.imshow(rgb)
    plt.axis("off")
    plt.subplot(132)
    # plt.title("pred")
    plt.imshow(labelviz_pred)
    plt.axis("off")
    plt.subplot(133)
    # plt.title("gt")
    plt.imshow(labelviz_gt)
    plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)

    img = imgviz.io.pyplot_to_numpy()
    plt.imsave(path, img)
    plt.close('all')
