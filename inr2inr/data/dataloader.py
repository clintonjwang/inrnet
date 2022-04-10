import os, torch
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image

from util import glob2

class cocoDS(torchvision.datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.paths = glob2(self.root,"*.jpg")

    def __getitem__(self, ix):
        return self.transform(Image.open(self.paths[ix]))
    def __len__(self):
        return len(self.paths)

def get_dataloader(args):
    paths, dl_args = (args["paths"], args["data loading"])
    if dl_args["dataset"] == "ImageNet":
        trans = transforms.Compose([transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder(
            "/data/vision/polina/scratch/clintonw/datasets/imagenet_pytorch/train",
            transform=trans)
        # imagenet_data = torchvision.datasets.ImageNet()
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=1, shuffle=False)

    elif dl_args["dataset"] == "COCO":
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = cocoDS(
            "/data/vision/polina/scratch/clintonw/datasets/coco/train2017",
            transform=trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # elif dl_args["dataset"] == "CelebA":
    #     trans = transforms.Compose([transforms.ToTensor(),
    #     ])
    #     dataset = torchvision.datasets.ImageFolder(
    #         "/data/vision/polina/scratch/clintonw/datasets/CelebA/train",
    #         transform=trans)
    #     data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    elif dl_args["dataset"] == "Flickr":
        trans = transforms.Compose([transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.Flickr30k(root="/data/vision/polina/scratch/clintonw/datasets/flickr",
            ann_file="/data/vision/polina/scratch/clintonw/datasets/flickr",
            transform=trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    elif dl_args["dataset"] == "CIFAR-100":
        trans = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = torchvision.datasets.CIFAR100(root="/data/vision/polina/scratch/clintonw/datasets/cifar",
                 transform=trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    else:
        raise NotImplementedError
    return data_loader
