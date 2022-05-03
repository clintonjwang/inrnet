import os, pdb, torch, argparse
osp = os.path
import numpy as np
import matplotlib.pyplot as plt
import monai.transforms as mtr

import args as args_module
import optim, util, losses
from data import dataloader
from models.inrs import siren

DATA_DIR = osp.expanduser("/data/vision/polina/scratch/clintonw/datasets/inrnet")

def train_siren(args):
    dl_args = args["data loading"]
    ds_name = dl_args["dataset"]
    interval = dl_args["interval"]
    keys = siren.get_siren_keys()
    param_dicts = {k:[] for k in keys}
    start_ix = int(args["start_ix"])
    end_ix = start_ix+dl_args["end_ix"]
    total_steps = args["optimizer"]["max steps"]

    while osp.exists(DATA_DIR+f"/{ds_name}/siren_{start_ix+interval-1}.pt"):
        start_ix += interval
    other_data = []
    print("Starting", flush=True)

    dataset = dataloader.get_img_dataset(args)
    for ix in range(start_ix,end_ix):
        data = dataset[ix]
        if isinstance(data, tuple):
            data = {'img':data[0], 'cls':data[1]}
        if data is None:
            continue
        if dl_args["variables"] is None:
            img = data
        else:
            img = data.pop("img")
            other_data.append(data)
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
        for k,v in inr.state_dict().items():
            param_dicts[k].append(v.cpu())

        if 'subset' in dl_args:
            path = DATA_DIR+f"/{ds_name}_{dl_args['subset']}/siren_{ix}.pt"
        else:
            path = DATA_DIR+f"/{ds_name}/siren_{ix}.pt"

        if ix % interval == interval-1:
            if dl_args["variables"] is None:
                torch.save(param_dicts, path)
            else:
                torch.save((param_dicts, other_data), path)
                other_data = []
            param_dicts = {k:[] for k in keys}

    print("Finished")

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

    if args["network"]["type"] == "SIREN":
        train_siren(args)
    else:
        raise NotImplementedError


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
