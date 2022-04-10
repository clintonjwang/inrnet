import torch
# import util, losses

# def update_SIT_weights(SIT_w, loss_args):
#     if "regularizer decay" in loss_args:
#         for k in SIT_w:
#             SIT_w[k] *= 1-loss_args["regularizer decay"]

# def get_kwargs(optimizer_settings):
#     kwargs = {"weight_decay":optimizer_settings["weight decay"]}
#     if optimizer_settings["type"] == "AdamW":
#         optim_class = torch.optim.AdamW
#     elif optimizer_settings["type"] == "Adam":
#         optim_class = torch.optim.Adam
#         kwargs["betas"] = (.5,.999)
#     elif optimizer_settings["type"] == "SGD":
#         optim_class = torch.optim.SGD
#         kwargs["momentum"] = optimizer_settings["momentum"]
#     return optim_class, kwargs

def get_optimizer(model, optimizer_settings):
    if optimizer_settings["type"] == "Adam":
        optim_class = torch.optim.Adam
    elif optimizer_settings["type"] == "AdamW":
        optim_class = torch.optim.AdamW
    else:
        raise NotImplementedError
    optimizer = optim_class(model.parameters(), lr=optimizer_settings["learning rate"],
        weight_decay=optimizer_settings["weight decay"])
    return optimizer

# def get_CAAE_optims(models, optimizer_settings):
#     optim_class, kwargs = get_kwargs(optimizer_settings)
#     G_optim = optim_class(models["G"].parameters(), lr=optimizer_settings["G learning rate"], **kwargs)
#     Dz_optim = optim_class(models["Dz"].parameters(), lr=optimizer_settings["Dz learning rate"], **kwargs)
#     Dimg_optim = optim_class(models["Dimg"].parameters(), lr=optimizer_settings["Dimg learning rate"], **kwargs)
#     return {'G':G_optim, 'Dz':Dz_optim, 'Dimg':Dimg_optim}

# def get_CVAE_optimizer(G, optimizer_settings):
#     optim_class, kwargs = get_kwargs(optimizer_settings)
#     G_optim = optim_class(G.parameters(), lr=optimizer_settings["G learning rate"], **kwargs)
#     return G_optim