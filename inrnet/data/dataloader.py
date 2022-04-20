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

def get_img_dataloader(args):
    paths, dl_args = (args["paths"], args["data loading"])
    if dl_args["dataset"] == "ImageNet":
        trans = transforms.Compose([transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder(
            DS_DIR+"/imagenet_pytorch/train",
            transform=trans)
        # imagenet_data = torchvision.datasets.ImageNet()
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=False)

    elif dl_args["dataset"] == "coco":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = jpgDS(DS_DIR+"/coco/train2017", transform=trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    elif dl_args["dataset"] == "horse":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = jpgDS(DS_DIR+"/inrnet/horse2zebra/trainA", transform=trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    elif dl_args["dataset"] == "zebra":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset = jpgDS(DS_DIR+"/inrnet/horse2zebra/trainB", transform=trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    elif dl_args["dataset"] == "kitti":
        data_loader = kitti.get_kitti_img_dataloader()

    else:
        raise NotImplementedError
    return data_loader


def get_inr_dataloader(dl_args):
    if dl_args["dataset"] == "kitti":
        return kitti.get_kitti_inr_dataloader()
    elif dl_args["dataset"] == "horse2zebra":
        return inr_loaders.get_h2z_inr_dataloader("horse"), inr_loaders.get_h2z_inr_dataloader("zebra")
    else:
        raise NotImplementedError

