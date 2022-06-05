"""INR classification"""
import os, pdb, torch, wandb
osp = os.path
nn = torch.nn
F = nn.functional
import numpy as np
import torchvision.models

# from inrnet import RESULTS_DIR
from inrnet import args as args_module, inn, util, jobs as job_mgmt
from inrnet.data import dataloader
from inrnet.inn import point_set
from inrnet.models.common import Conv2, Conv5
from inrnet.inn.nets.effnet import InrCls2, InrCls4, InrClsWide2, InrClsWide4

def train_classifier(args):
    wandb.init(project="inrnet", job_type="train", name=args["job_id"],
        config=wandb.helper.parse_config(args, exclude=['job_id', 'start_ix']))#, 'paths'],
    args = args_module.get_wandb_config()
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    val_data_loader = dataloader.get_val_inr_dataloader(dl_args)
    global_step = 0
    loss_fxn = nn.CrossEntropyLoss()
    top3 = lambda pred_cls, labels: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()
    top1 = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()

    model = load_pretrained_model(args).cuda()
    optimizer = util.get_optimizer(model, args)
    
    if hasattr(model, 'layers'):
        wandb.watch(model.layers[0][0])
        wandb.watch(model.layers[2][0])
    else:
        wandb.watch(model[0][0])
        wandb.watch(model[2][0])
        
    for img_inr, labels in data_loader:
        global_step += 1

        if args["network"]['type'].startswith('inr'):
            logit_fxn = model(img_inr)
            logit_fxn.change_sample_mode(dl_args['sample type'])
            coords = point_set.generate_sample_points(logit_fxn, dl_args)
            logits = logit_fxn(coords)
        else:
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img)

        loss = loss_fxn(logits, labels)
        pred_cls = logits.topk(k=3).indices
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        wandb.log({'train loss':loss.item(),
            'top 3 acc': top3(pred_cls, labels).item(),
            'top 1 acc': top1(pred_cls, labels).item(),
        })
        if global_step % 100 == 0:
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "best.pth"))
        if global_step >= args["optimizer"]["max steps"]:
            break
        if global_step % 2 == 0:
            with torch.no_grad():
                img_inr, labels = next(val_data_loader)
                if args["network"]['type'].startswith('inr'):
                    logit_fxn = model(img_inr).eval()
                    logit_fxn.change_sample_mode(dl_args['sample type'])
                    coords = point_set.generate_sample_points(logit_fxn, dl_args)
                    logits = logit_fxn(coords)

                else:
                    img = img_inr.produce_images(*dl_args['image shape'])
                    logits = model(img)

                loss = loss_fxn(logits, labels)
                pred_cls = logits.topk(k=3).indices
                wandb.log({'val loss':loss.item(),
                    'val top 3 acc': top3(pred_cls, labels).item(),
                    'val top 1 acc': top1(pred_cls, labels).item(),
                }, step=global_step)
            
    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))




def test_inr_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    top3, top1 = 0,0
    origin = args['target_job']
    model = load_model_from_job(origin).cuda().eval()
    orig_args = job_mgmt.get_job_args(origin)

    with torch.no_grad():
        for img_inr, labels in data_loader:
            if orig_args["network"]['type'].startswith('inr'):
                logit_fxn = model(img_inr).cuda().eval()
                logit_fxn.change_sample_mode(dl_args['sample type'])
                coords = point_set.generate_sample_points(logit_fxn, dl_args)
                logits = logit_fxn(coords)
            else:
                img = img_inr.produce_images(*dl_args['image shape'])
                logits = model(img)

            pred_cls = logits.topk(k=3).indices
            top3 += (labels.unsqueeze(1) == pred_cls).amax(1).float().sum().item()
            top1 += (labels == pred_cls[:,0]).float().sum().item()

    torch.save((top1, top3), osp.join(paths["job output dir"], "stats.pt"))




# import pandas as pd

# def train_material_classifier(args):
#     paths = args["paths"]
#     df = pd.read_csv(osp.expanduser('~/code/ObjectFolder/objects300.csv'), index_col=0)
#     df = df[df['material'].isin(('Ceramic', 'Polycarbonate', 'Wood'))]
#     materials = df['material']
#     classes, counts = np.unique(materials.values, return_counts=True)
#     loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())
#     top1 = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()
#     top1_tracker = util.MetricTracker("top1", function=top1)

#     model = load_pretrained_model(args).cuda()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])

#     while True:
#         for i in df.index:
#             N = dl_args["sample points"]
#             global_step += 1
#             logit_fxn = model(img_inr)
#             coords = logit_fxn.generate_sample_points(sample_size=N, method='rqmc')
#             logits = logit_fxn(coords)

#             loss = loss_tracker(logits, labels)
#             pred_cls = logits.topk(k=5).indices
#             top_5 = top3_tracker(pred_cls, labels).item()
#             top_1 = top1_tracker(pred_cls, labels).item()

#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()

#             if global_step % 20 == 0:
#                 print(np.round(loss.item(), decimals=3), "; top_5:", np.round(top_5, decimals=2),
#                     "; top_1:", np.round(top_1, decimals=2),
#                     flush=True)
#             if global_step % 100 == 0:
#                 torch.save(model.state_dict(), osp.join(paths["weights dir"], "best.pth"))
#                 loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
#                 top1_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top1.png")
#                 top3_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top3.png")

#             if global_step >= args["optimizer"]["max steps"]:
#                 break

#     torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))



def load_pretrained_model(args):
    net_args = args["network"]
    in_ch = args['data loading']['input channels']
    n_classes = args['data loading']['classes']
    kwargs = dict(in_channels=in_ch, out_dims=n_classes)
    if net_args["type"].startswith("inr"):
        kwargs['mid_ch'] = net_args['conv']['mid_ch']
        kwargs['scale2'] = net_args['conv']['scale2']
    if net_args["type"] == "cnn-2":
        return Conv2(**kwargs)
    elif net_args["type"] == "cnn-5":
        return Conv5(**kwargs)
    elif net_args["type"] == "inr-2":
        return InrCls2(**kwargs)
    elif net_args["type"] == "inr-w2":
        return InrClsWide2(**kwargs)
    elif net_args["type"] == "inr-5":
        return InrCls4(**kwargs)
    elif net_args["type"] == "inr-w5":
        return InrClsWide4(**kwargs)

    pretrained = net_args['pretrained']
    img_shape = args["data loading"]["image shape"]
    if isinstance(pretrained, str):
        base = load_model_from_job(pretrained)
    else:
        out = nn.Linear(24, args["data loading"]['classes'])
        nn.init.kaiming_uniform_(out.weight, mode='fan_in')
        out.bias.data.zero_()
        if net_args["type"] == "effnet-s3":
            m = torchvision.models.efficientnet_b0(pretrained=False)
            return nn.Sequential(m.features[:3],
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(1), out)

        elif net_args["type"] == "resnet":
            out = nn.Linear(256, args["data loading"]['classes'])
            nn.init.kaiming_uniform_(out.weight, mode='fan_in')
            out.bias.data.zero_()
            m = torchvision.models.resnet18(pretrained=False)
            return nn.Sequential(
                m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3,
                nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(1), out)

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
    path = osp.expanduser(f"~/code/inrnet/results/{origin}/weights/best.pth")
    model = load_pretrained_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

