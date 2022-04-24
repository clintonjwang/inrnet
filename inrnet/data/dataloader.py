import os, torch
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
import dill as pickle

from inrnet.util import glob2
from inrnet.data import kitti, inr_loaders

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
        return {"cls":self.classes.index(osp.basename(osp.dirname(self.subpaths[ix]))),
        "img":self.transform(Image.open(osp.join(self.root, self.subpaths[ix])))}
    def __len__(self):
        return len(self.subpaths)

def get_img_dataset(args):
    dl_args = args["data loading"]
    if dl_args["dataset"] == "imagenet1k":
        trans = transforms.Compose([
            transforms.Resize((224,224)),
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
        return inr_loaders.get_h2z_inr_dataloader("horse"), inr_loaders.get_h2z_inr_dataloader("zebra")
    else:
        raise NotImplementedError

