import os, torch
osp=os.path
from inspect import signature

TMP_DIR=osp.expanduser("~/code/diffcoord/temp")

def inference(model, imgs, args):
    with torch.cuda.amp.autocast():
        if args["network"]["type"].lower() == "unetr":
            pred_logits = sliding_window_inference(imgs, args["network"]["window size"], 4, model)
        elif args["network"]["type"].lower() == "topnet":
            pred_segs = topnet_inference(model, imgs, args)
        else:
            pred_logits = model(imgs)
    return pred_logits


def get_model(paths, args):
    net_args=args["network"]
    if net_args["type"] == "U-Net":
        model = UNet()
    elif net_args["type"] == "TopNet":
        model = TopNet(C=net_args["min channels"], mini=args["data loading"]["dataset"] == "synthetic")
    elif net_args["type"].lower() == "unetr":
        params = signature(UNETR.__init__).parameters
        kwargs = {k:v for k,v in net_args.items() if k in params.keys()}
        model = UNETR(
            in_channels=1,
            out_channels=1,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            img_size=net_args["window size"],
            **kwargs,
        )
        model.vit.patch_embedding.cls_token = None
        
    else:
        raise NotImplementedError

    load_pretrained_model(model, paths)
    return model.cuda().eval()


def load_pretrained_model(model, paths):
    if paths["pretrained model name"] is not None:
        init_weight_path = osp.join(TMP_DIR, paths["pretrained model name"])
        if not osp.exists(init_weight_path):
            raise ValueError(f"bad pretrained model path {init_weight_path}")

        checkpoint_sd = torch.load(init_weight_path)
        model_sd = model.state_dict()
        for k in model_sd.keys():
            if k in checkpoint_sd.keys() and checkpoint_sd[k].shape != model_sd[k].shape:
                checkpoint_sd.pop(k)

        model.load_state_dict(checkpoint_sd, strict=False)
