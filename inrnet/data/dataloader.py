import os, torch, pdb
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from inrnet.models.inrs import siren
from inrnet.util import glob2
from inrnet.data import kitti, inet, cityscapes

DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

class jpgDS(torchvision.datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.paths = glob2(self.root,"*.jpg")
    def __getitem__(self, ix):
        return self.transform(Image.open(self.paths[ix]))
    def __len__(self):
        return len(self.paths)

def get_img_dataset(args):
    totorch_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dl_args = args["data loading"]
    if dl_args["dataset"] == "imagenet1k":
        dataset = inet.INetDS(DS_DIR+"/imagenet_pytorch", subset=dl_args['subset'])

    elif dl_args["dataset"] == "kitti":
        raise NotImplementedError("kitti dataset")
        data_loader = kitti.get_kitti_img_dataloader()

    elif dl_args["dataset"] == "coco":
        dataset = jpgDS(DS_DIR+"/coco/train2017", transform=transforms.ToTensor())

    elif dl_args["dataset"] == "horse":
        dataset = jpgDS(DS_DIR+"/inrnet/horse2zebra/trainA", transform=transforms.ToTensor())

    elif dl_args["dataset"] == "zebra":
        dataset = jpgDS(DS_DIR+"/inrnet/horse2zebra/trainB", transform=transforms.ToTensor())

    elif dl_args["dataset"] == "places_std":
        dataset = torchvision.datasets.ImageFolder(
            DS_DIR+"/places365_standard/val", transform=totorch_norm)

    elif dl_args["dataset"] == "cifar10":
        return torchvision.datasets.CIFAR10(root=DS_DIR, train=dl_args['subset'] == 'train',
            transform=transforms.ToTensor())

    elif dl_args["dataset"] == "cityscapes":
        size = dl_args['image shape']
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return torchvision.datasets.Cityscapes(DS_DIR+'/cityscapes',
            split=dl_args['subset'], mode='coarse', target_type='semantic',
            transform=trans, target_transform=cityscapes.seg_transform)

    elif dl_args["dataset"] == "flowers":
        size = dl_args['image shape']
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
        ])
        return torchvision.datasets.Flowers102(DS_DIR,
            split=dl_args['subset'], transform=trans)

    else:
        raise NotImplementedError
    return dataset


def get_inr_dataloader(dl_args):
    if dl_args["dataset"] == "kitti":
        return kitti.get_kitti_inr_dataloader()
    elif dl_args["dataset"] == "horse2zebra":
        return get_inr_loader_for_imgds("horse"), get_inr_loader_for_imgds("zebra")
    elif dl_args["dataset"] in ("imagenet1k", 'cifar10'):
        return get_inr_loader_for_cls_ds(dl_args["dataset"],
            bsz=dl_args['batch size'], subset=dl_args['subset'])
    elif dl_args["dataset"] == "inet12":
        return inet.get_inr_loader_for_inet12(bsz=dl_args['batch size'], subset=dl_args['subset'])
    else:
        raise NotImplementedError(dl_args["dataset"])

def get_inr_loader_for_cls_ds(ds_name, bsz, subset):
    inr = siren.Siren(out_channels=3)
    paths = glob2(f"{DS_DIR}/inrnet/{ds_name}_{subset}/siren_*.pt")
    keys = siren.get_siren_keys()
    def random_loader():
        while True:
            np.random.shuffle(paths)
            for path in paths:
                data, classes = torch.load(path)
                indices = list(range(len(data)))
                ix = np.random.choice(indices)
                param_dict = {k:data[k][ix] for k in keys}
                try:
                    inr.load_state_dict(param_dict)
                except RuntimeError:
                    continue
                yield inr.cuda(), torch.tensor(classes[ix]['cls']).unsqueeze(0)
    return random_loader()

    
def get_inr_loader_for_imgds(dataset):
    inr = siren.Siren(out_channels=3)
    paths = sorted(glob2(f"{DS_DIR}/inrnet/{dataset}/siren_*.pt"))
    keys = siren.get_siren_keys()
    for path in paths:
        data = torch.load(path)
        for ix in range(len(data)):
            param_dict = {k:data[k][ix] for k in keys}
            try:
                inr.load_state_dict(param_dict)
            except RuntimeError:
                continue
            yield inr.cuda()

