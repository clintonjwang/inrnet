import os, pdb, gc, torch
osp = os.path
import numpy as np
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance

import args as args_module
import optim, util, losses
from data import process_data
from data.dataloader import get_dataloaders
from models import model as model_module
torch.backends.cudnn.benchmark = True


def pretrain_masked_autoencoder(args, dataloaders):
    paths=args["paths"]
    loss_args=args["loss"]
    optim_args=args["optimizer"]
    net_args=args["network"]

    model = model_module.get_model(paths, net_args).train()
    optimizer = optim.get_optimizer(model, optim_args)
    gradscaler = torch.cuda.amp.GradScaler()

    global_step = 0
    
    segloss_tracker = util.MetricTracker(name="segmentation loss", function=losses.get_seg_loss(loss_args))
    metric_trackers = [segloss_tracker, reg_tracker]
    
    def backward(loss, optimizer):
        optimizer.zero_grad()
        gradscaler.scale(loss).backward()
        gradscaler.step(optimizer)
        gradscaler.update()

    max_epochs = optim_args["epochs"]
    for epoch in range(1,max_epochs+1):
        epoch_iterator = tqdm(
            dataloaders["train"], desc="Training (Step X, Epoch X/X) (loss=X.X)", dynamic_ncols=True
        )
        for batch in epoch_iterator:
            global_step += 1
            imgs, gt_segs = batch["image"].cuda(), batch["label"].cuda()
            with torch.cuda.amp.autocast():
                pred_logits = model(imgs)
            loss = segloss_tracker(pred_logits, gt_segs, phase="train")
            backward(loss, optimizer)
            epoch_iterator.set_description(
                "Training (Step %d, Epoch %d/%d) (loss=%2.3f)" % (global_step, epoch, max_epochs, loss.item())
            )
            if epoch % eval_interval == 0:
                with torch.no_grad():
                    dice = dice_tracker(pred_logits, gt_segs, phase="train")

        if loss_tracker.is_at_min("train"):
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            util.save_example(epoch, paths["job output dir"]+"/imgs/val",
                imgs[0], gt_segs[0], pred_logits[0])

            if epoch % hausdorff_interval == 0:
                util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")
                if hausdorff_tracker.is_at_min():
                    torch.save(model.state_dict(), osp.join(paths["weights dir"], "best_hausdorff.pth"))
        
        for t in metric_trackers:
            t.update_at_epoch_end()

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))
