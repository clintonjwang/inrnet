import os, pdb, torch
osp = os.path
import numpy as np

import args as args_module
import optim, util, losses
from data import dataloader
from models import model as model_module, common, img2d

TMP_DIR = osp.expanduser("~/code/diffcoord/temp")

def train_inr(args, data_loader):
    keys = ('net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight', 'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight', 'net.3.linear.bias', 'net.4.weight', 'net.4.bias')
    param_dicts = {k:[] for k in keys}
    start_ix=args["data loading"]["start_ix"]
    for ix,img in enumerate(data_loader):
        if img.size(1) == 1 or ix < start_ix:
            continue
        imgFit = img2d.ImageFitting(img)
        H,W = imgFit.H,imgFit.W
        dl = torch.utils.data.DataLoader(imgFit, batch_size=1, pin_memory=True, num_workers=0)
        INR = img2d.Siren().cuda()
        total_steps = 500
        optim = torch.optim.Adam(lr=1e-4, params=INR.parameters())
        model_input, ground_truth = next(iter(dl))
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        for step in range(total_steps):
            model_output, coords = INR(model_input)    
            loss = (losses.mse_loss(model_output, ground_truth)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if loss.item() < .02:
                break
        print("Loss %0.4f" % (loss))
        for k,v in INR.state_dict().items():
            param_dicts[k].append(v.cpu())
        
        if ix % 64 == 63:
            torch.save(param_dicts, TMP_DIR+f"/siren_{ix}.pt")
            param_dicts = {k:[] for k in keys}
    print("finished")

def train_diffusion_model():
    # paths=args["paths"]
    # loss_args=args["loss"]
    # optim_args=args["optimizer"]
    # model = model_module.get_model(paths, args).train()
    # optimizer = optim.get_optimizer(model, optim_args)
    # gradscaler = torch.cuda.amp.GradScaler()

    # eval_interval = 20
    # global_step = 0
    
    # intervals = {"train":1, "val":eval_interval}
    # select_inds = np.random.choice(
    #     coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
    # )
    max_epochs = optim_args["epochs"]
    for epoch in range(1,max_epochs+1):
        for imgs,_ in data_loader:
            global_step += 1
            imgs, gt_segs = batch.cuda()
            with torch.cuda.amp.autocast():
                loss = loss_tracker(recon, sample_rgb, phase="train")
            backward(loss, optimizer)

        print("epoch:", epoch, "| train loss:", segloss_tracker.epoch_average())
        util.save_example(paths["job output dir"]+f"/imgs/train/{epoch}.png",
            imgs[0], gt_segs[0], pred_logits[0])

        if epoch % eval_interval == 0 or epoch == max_epochs:
            with torch.no_grad():
                model.eval()
                for batch in dl["val"]:
                    imgs, gt_segs = batch["image"].cuda(), batch["label"].cuda()
                    pred_logits = model_module.inference(model, imgs, args=args)
                    segloss_tracker(pred_logits, gt_segs, phase="val")
                    if regularizer is not None:
                        regularizer(pred_logits, gt_segs, phase="val")

                    dice_tracker(pred_logits, gt_segs)
                    if epoch % hausdorff_interval == 0:
                        hausdorff_tracker(pred_logits > 0, gt_segs)
                        components_tracker(pred_logits, gt_segs)
                model.train()

            if dice_tracker.is_at_max():
                torch.save(model.state_dict(), osp.join(paths["weights dir"], "best_dice.pth"))
                # util.save_metric_histograms(metric_trackers, root=paths["job output dir"]+"/plots")
            print("Best Avg. Dice: %.3f Current Avg. Dice: %.3f" % (dice_tracker.max(), dice_tracker.epoch_average()))
            util.save_example(paths["job output dir"]+f"/imgs/val/{epoch}.png",
                imgs[0], gt_segs[0], pred_logits[0])

            if epoch % hausdorff_interval == 0:
                util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")
                if hausdorff_tracker.is_at_min():
                    torch.save(model.state_dict(), osp.join(paths["weights dir"], "best_hausdorff.pth"))
        
        for t in metric_trackers:
            t.update_at_epoch_end()

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def main(args):
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    data_loader = dataloader.get_dataloader(args)
    if args["network"]["type"] == "INR":
        train_inr(args, data_loader=data_loader)

if __name__ == "__main__":
    args = args_module.parse_args()
    main(args)