import os, pdb, torch, argparse
osp = os.path
import numpy as np
import matplotlib.pyplot as plt
import monai.transforms as mtr

import args as args_module
from inrnet import optim, util, losses
from inrnet.data import dataloader
from inrnet.models.inrs import siren
from inrnet.data import inet

DATA_DIR = osp.expanduser("/data/vision/polina/scratch/clintonw/datasets/inrnet")

def fit_img_to_siren(img, total_steps):
    imgFit = siren.ImageFitting(img.unsqueeze_(0))
    H,W = imgFit.H, imgFit.W
    dl = torch.utils.data.DataLoader(imgFit, batch_size=1, pin_memory=True, num_workers=0)
    inr = siren.Siren(out_channels=img.size(1)).cuda()
    optim = torch.optim.Adam(lr=1e-4, params=inr.parameters())
    model_input, ground_truth = next(iter(dl))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(total_steps):
        model_output = inr(model_input)    
        loss = (losses.mse_loss(model_output, ground_truth)).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("Loss %0.4f" % (loss), flush=True)
    return inr, loss


def train_siren(args):
    dl_args = args["data loading"]
    ds_name = dl_args["dataset"]
    keys = siren.get_siren_keys()
    start_ix = int(args["start_ix"])
    end_ix = start_ix+dl_args["end_ix"]
    total_steps = args["optimizer"]["max steps"]

    while osp.exists(DATA_DIR+f"/{ds_name}/{dl_args['subset']}_{start_ix}.pt"):
        start_ix += 1
    print("Starting", flush=True)

    dataset = dataloader.get_img_dataset(args)
    param_dict = {}
    for ix in range(start_ix,end_ix):
        data = dataset[ix]
        if isinstance(data, tuple):
            img, data = data[0], data[1:]
        else:
            img = data

        inr,loss = fit_img_to_siren(img, total_steps)
        for k,v in inr.state_dict().items():
            param_dict[k] = v.cpu()

        path = DATA_DIR+f"/{ds_name}/{dl_args['subset']}_{ix}.pt"
        loss_path = DATA_DIR+f"/{ds_name}/loss_{dl_args['subset']}_{ix}.txt"
        open(loss_path, 'w').write(str(loss.item()))

        if dl_args["variables"] is None:
            torch.save(param_dict, path)
        else:
            torch.save((param_dict, data[1:]), path)


    # import monai.transforms as mtr
    # rescale_float = mtr.ScaleIntensity()
    # fname = osp.expanduser("~/downloads/orig.png")
    # arr = rescale_float(img.squeeze().permute(1,2,0).cpu().numpy())
    # plt.imsave(fname, arr)
    # inr.H,inr.W=H,W
    # arr = rescale_float(inr.produce_image().cpu().numpy())
    # fname = osp.expanduser("~/downloads/siren.png")
    # plt.imsave(fname, arr)

def main(args):
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    if args["data loading"]["dataset"] == "inet12":
        train_inet12(args)
    elif args["data loading"]["dataset"] == "cityscapes":
        train_cityscapes(args)
    elif args["data loading"]["dataset"] == "flowers":
        train_flowers(args)
        
    elif args["network"]["type"] == "SIREN":
        train_siren(args)
    else:
        raise NotImplementedError


def train_inet12(args):
    dl_args = args["data loading"]
    keys = siren.get_siren_keys()
    start_ix = int(args["start_ix"])
    # cls, start_ix = start_ix // 1000, start_ix % 1000
    # if start_ix < 800:
    #     subset='train'
    # else:
    #     subset='test'
    #     start_ix -= 800

    end_ix = start_ix+100
    subset='train'
    total_steps = args["optimizer"]["max steps"]

    # os.makedirs(DATA_DIR+f"/inet12/{cls}", exist_ok=True)
    # while osp.exists(DATA_DIR+f"/inet12/{cls}/{subset}_{start_ix}.pt"):
    #     start_ix += 1
    print("Starting", flush=True)

    dataset = inet.INet12Extra()
    # dataset = inet.INet12(subset=subset, cls=cls)
    param_dict = {}
    for j in range(start_ix,end_ix):
        cls, ix = j % 12, j // 12 + 800
        path = DATA_DIR+f"/inet12/{cls}/{subset}_{ix}.pt"
        while osp.exists(path):
            continue

        img = dataset[j]
        inr,loss = fit_img_to_siren(img, total_steps)
        for k,v in inr.state_dict().items():
            param_dict[k] = v.cpu()

        torch.save(param_dict, path)
        loss_path = DATA_DIR+f"/inet12/{cls}/loss_{subset}_{ix}.txt"
        open(loss_path, 'w').write(str(loss.item()))


def train_flowers(args):
    args["start_ix"] = int(args["start_ix"])
    if args["start_ix"] < 1020:
        args["data loading"]['subset']='train'
    elif args["start_ix"] < 2040:
        args["data loading"]['subset']='val'
        args["start_ix"] -= 1020
    else:
        args["data loading"]['subset']='test'
        args["start_ix"] -= 2040
    return train_siren(args)


def train_cityscapes(args):
    dl_args = args["data loading"]
    keys = siren.get_siren_keys()
    start_ix = int(args["start_ix"])
    if start_ix < 2975:
        dl_args['subset']='train'
    else:
        dl_args['subset']='val'
        start_ix -= 2975
    end_ix = start_ix+100
    total_steps = args["optimizer"]["max steps"]
    subset=dl_args['subset']

    os.makedirs(DATA_DIR+f"/cityscapes_nonorm", exist_ok=True)
    while osp.exists(DATA_DIR+f"/cityscapes_nonorm/{subset}_{start_ix}.pt"):
        start_ix += 1
    print("Starting", flush=True)

    dataset = dataloader.get_img_dataset(args)
    param_dict = {}
    for ix in range(start_ix,end_ix):
        img,seg = dataset[ix]
        inr,loss = fit_img_to_siren(img, total_steps)
        for k,v in inr.state_dict().items():
            param_dict[k] = v.cpu()

        path = DATA_DIR+f"/cityscapes_nonorm/{subset}_{ix}.pt"
        torch.save((param_dict, seg.squeeze(0)), path)
        loss_path = DATA_DIR+f"/cityscapes_nonorm/loss_{subset}_{ix}.txt"
        open(loss_path, 'w').write(str(loss.item()))


if __name__ == "__main__":
    # from data import kitti
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config_name')
    # parser.add_argument('-j', '--job_id', default="manual")
    # parser.add_argument('-s', '--start_ix', default=0, type=int)
    # args = parser.parse_args()
    # print(args.start_ix)
    # kitti.save_kitti_imgs(args.start_ix)
    args = args_module.parse_args()
    main(args)
