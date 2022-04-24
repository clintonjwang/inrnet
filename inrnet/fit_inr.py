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
    dataset = dataloader.get_img_dataset(args)
    keys = ('net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight', 'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight', 'net.3.linear.bias', 'net.4.weight', 'net.4.bias')
    param_dicts = {k:[] for k in keys}
    start_ix=int(args["start_ix"])
    end_ix = start_ix+args["data loading"]["end_ix"]
    ds_name = args["data loading"]["dataset"]
    interval = args["data loading"]["interval"]
    total_steps = args["optimizer"]["max steps"]

    while osp.exists(DATA_DIR+f"/{ds_name}/siren_{start_ix+interval-1}.pt"):
        start_ix += interval
    other_data = []
    print("Starting", flush=True)
    for ix in range(start_ix,end_ix):
        data = dataset[ix]

        if args["data loading"]["variables"] is None:
            img = data
        else:
            img = data.pop("img")
            other_data.append(data)
        imgFit = siren.ImageFitting(img.unsqueeze_(0))
        H,W = imgFit.H, imgFit.W
        dl = torch.utils.data.DataLoader(imgFit, batch_size=1, pin_memory=True, num_workers=0)
        INR = siren.Siren(out_channels=img.size(1)).cuda()
        optim = torch.optim.Adam(lr=1e-4, params=INR.parameters())
        model_input, ground_truth = next(iter(dl))
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        for step in range(total_steps):
            model_output = INR(model_input)    
            loss = (losses.mse_loss(model_output, ground_truth)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        print("Loss %0.4f" % (loss), flush=True)
        for k,v in INR.state_dict().items():
            param_dicts[k].append(v.cpu())
        
        if ix % interval == interval-1:
            if args["data loading"]["variables"] is None:
                torch.save(param_dicts, DATA_DIR+f"/{dataset}/siren_{ix}.pt")
            else:
                torch.save((param_dicts, other_data), DATA_DIR+f"/{dataset}/siren_{ix}.pt")
                other_data = []
            param_dicts = {k:[] for k in keys}
    print("finished")

        # import monai.transforms as mtr
        # rescale_float = mtr.ScaleIntensity()
        # fname = osp.expanduser("~/downloads/orig.png")
        # arr = rescale_float(img.squeeze().permute(1,2,0).cpu().numpy())
        # plt.imsave(fname, arr)
        # INR.H,INR.W=H,W
        # arr = rescale_float(INR.produce_image().cpu().numpy())
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
