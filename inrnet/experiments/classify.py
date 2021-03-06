"""INR classification"""
import os
import pdb
import torch
import wandb

from inrnet.inn.nets.consistent import LearnedSampler
from inrnet.utils import args as args_module, jobs as job_mgmt, util
osp = os.path
nn = torch.nn
F = nn.functional
import torchvision.models

# from inrnet import RESULTS_DIR
from inrnet import inn
from inrnet.data import dataloader
from inrnet.inn import point_set
from inrnet.models.common import Conv2, Conv5
from inrnet.inn.nets.classifier import InrCls

def train_classifier(args: dict) -> None:
    if not args['no_wandb']:
        wandb.init(project="inrnet", job_type="train", name=args["job_id"],
            config=wandb.helper.parse_config(args, exclude=['job_id']))
        args = args_module.get_wandb_config()
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    val_data_loader = dataloader.get_val_inr_dataloader(dl_args)
    global_step = 0
    loss_fxn = nn.CrossEntropyLoss()
    top3 = lambda pred_cls, labels: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()
    top1 = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()

    model = load_model(args).cuda()
    optimizer = util.get_optimizer(model, args)

    # ratio = 16
    # dense_sampler = model.sampler.copy()
    # dense_sampler['sample points'] *= ratio/4
    # query_layers = nn.Sequential()
    # flow_layers = nn.Sequential()
    # LearnedSampler(dense_sampler, query_layers, flow_layers, ratio=ratio)
    
    if not args['no_wandb']:
        if hasattr(model, 'layers'):
            wandb.watch(model.layers[0][0], log_freq=300)
        else:
            wandb.watch(model[0][0], log_freq=300)
        

    for img_inr, labels in data_loader:
        model.train()
        global_step += 1

        if args["network"]['type'].startswith('inr'):
            logits = model(img_inr)
        else:
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img)

        loss = loss_fxn(logits, labels)
        pred_cls = logits.topk(k=3).indices
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        log({'train_loss':loss.item(),
            'train_t3_acc': top3(pred_cls, labels).item(),
            'train_t1_acc': top1(pred_cls, labels).item(),
        }, use_wandb=not args['no_wandb'])
        
        if global_step % 100 == 0:
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "best.pth"))
        if global_step >= args["optimizer"]["max steps"]:
            break
        if global_step % 2 == 0:
            with torch.no_grad():
                img_inr, labels = next(val_data_loader)
                if args["network"]['type'].startswith('inr'):
                    model.eval()
                    logits = model(img_inr)

                else:
                    img = img_inr.produce_images(*dl_args['image shape'])
                    logits = model(img)

                loss = loss_fxn(logits, labels)
                pred_cls = logits.topk(k=3).indices
                log({'val_loss':loss.item(),
                        'val_t3_acc': top3(pred_cls, labels).item(),
                        'val_t1_acc': top1(pred_cls, labels).item(),
                    }, step=global_step, use_wandb=not args['no_wandb'])
            
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
        model.eval()
        for img_inr, labels in data_loader:
            if orig_args["network"]['type'].startswith('inr'):
                logits = model(img_inr)
            else:
                img = img_inr.produce_images(*dl_args['image shape'])
                logits = model(img)

            pred_cls = logits.topk(k=3).indices
            top3 += (labels.unsqueeze(1) == pred_cls).amax(1).float().sum().item()
            top1 += (labels == pred_cls[:,0]).float().sum().item()

    torch.save((top1, top3), osp.join(paths["job output dir"], "stats.pt"))

def log(variable, *args, use_wandb=True, **kwargs) -> None:
    if use_wandb:
        wandb.log(variable, *args, **kwargs)
    else:
        print(variable)

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

#     model = load_model(args).cuda()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])

#     while True:
#         for i in df.index:
#             N = dl_args["sample points"]
#             global_step += 1
#             logit_fxn = model(img_inr)
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



def load_model(args):
    net_args = args["network"]
    in_ch = args['data loading']['input channels']
    n_classes = args['data loading']['classes']
    kwargs = dict(in_channels=in_ch, out_dims=n_classes)
    if net_args["type"] == "cnn-2":
        return Conv2(**kwargs)
    elif net_args["type"] == "cnn-5":
        return Conv5(**kwargs)
    elif net_args["type"] == "inrnet":
        sampler = point_set.get_sampler_from_args(args['data loading'])
        return InrCls(sampler=sampler, **kwargs, **net_args['conv'])
    else:
        raise NotImplementedError

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
    model = load_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model
    