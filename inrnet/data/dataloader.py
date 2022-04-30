import os, torch, pdb
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
import dill as pickle
import numpy as np

from inrnet.models.inrs import siren
from inrnet.util import glob2
from inrnet.data import kitti

DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

class jpgDS(torchvision.datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.paths = glob2(self.root,"*.jpg")
    def __getitem__(self, ix):
        return self.transform(Image.open(self.paths[ix]))
    def __len__(self):
        return len(self.paths)

class INetDS(torchvision.datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.subpaths = open(self.root+"/val.txt", "r").read().split('\n')
        classes = open(self.root+"/labels.txt", 'r').read().split('\n')
        self.classes = [c[:c.find(',')] for c in classes]
    def __getitem__(self, ix):
        try:
            return {"cls":self.classes.index(osp.basename(osp.dirname(self.subpaths[ix]))),
                "img":self.transform(Image.open(osp.join(self.root, self.subpaths[ix])))}
        except RuntimeError:
            return None
    def __len__(self):
        return len(self.subpaths)

def get_img_dataset(args):
    dl_args = args["data loading"]
    if dl_args["dataset"] == "imagenet1k":
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = INetDS(DS_DIR+"/imagenet_pytorch", transform=trans)

    elif dl_args["dataset"] == "coco":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = jpgDS(DS_DIR+"/coco/train2017", transform=trans)

    elif dl_args["dataset"] == "horse":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = jpgDS(DS_DIR+"/inrnet/horse2zebra/trainA", transform=trans)
    elif dl_args["dataset"] == "zebra":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = jpgDS(DS_DIR+"/inrnet/horse2zebra/trainB", transform=trans)

    elif dl_args["dataset"] == "places_std":
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = torchvision.datasets.ImageFolder(
            DS_DIR+"/places365_standard/val", transform=trans)

    elif dl_args["dataset"] == "kitti":
        raise NotImplementedError("kitti dataset")
        data_loader = kitti.get_kitti_img_dataloader()

    else:
        raise NotImplementedError
    return dataset


def get_inr_dataloader(dl_args):
    if dl_args["dataset"] == "kitti":
        return kitti.get_kitti_inr_dataloader()
    elif dl_args["dataset"] == "horse2zebra":
        return get_inr_loader_for_imgds("horse"), get_inr_loader_for_imgds("zebra")
    elif dl_args["dataset"] == "imagenet1k":
        return get_inr_loader_for_cls_ds("imagenet1k")
    else:
        raise NotImplementedError

def get_inr_loader_for_cls_ds(dataset):
    inr = siren.Siren(out_channels=3)
    paths = sorted(glob2(f"{DS_DIR}/inrnet/{dataset}/siren_*.pt"))
    keys = ['net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight', 'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight', 'net.3.linear.bias', 'net.4.weight', 'net.4.bias']
    def ordered_loader():
        for path in paths:
            data, classes = torch.load(path)
            for ix in range(len(data)):
                param_dict = {k:data[k][ix] for k in keys}
                try:
                    inr.load_state_dict(param_dict)
                except RuntimeError:
                    continue
                yield inr.cuda(), torch.tensor(classes[ix]['cls']).unsqueeze(0)
    # dataset = 
    raise NotImplementedError("inr loader is not randomized")
    return torch.utils.data.DataLoader(dataset)

class ClsDS(torch.utils.data.IterableDataset):
    pass
    
def get_inr_loader_for_imgds(dataset):
    inr = siren.Siren(out_channels=3)
    paths = sorted(glob2(f"{DS_DIR}/inrnet/{dataset}/siren_*.pt"))
    keys = ['net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight', 'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight', 'net.3.linear.bias', 'net.4.weight', 'net.4.bias']
    for path in paths:
        data = torch.load(path)
        for ix in range(len(data)):
            param_dict = {k:data[k][ix] for k in keys}
            try:
                inr.load_state_dict(param_dict)
            except RuntimeError:
                continue
            yield inr.cuda()

