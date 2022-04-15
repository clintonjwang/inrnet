import torch

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
