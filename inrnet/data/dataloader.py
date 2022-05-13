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
    totorch_resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
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
            DS_DIR+"/places365_standard/val", transform=transforms.ToTensor())

    elif dl_args["dataset"] == "cifar10":
        return torchvision.datasets.CIFAR10(root=DS_DIR, train=dl_args['subset'] == 'train',
            transform=transforms.ToTensor())

    elif dl_args["dataset"] == "cityscapes":
        size = dl_args['image shape']
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if dl_args['subset'] == 'train_extra':
            mode = 'coarse'
        else:
            mode = 'fine'
        return torchvision.datasets.Cityscapes(DS_DIR+'/cityscapes',
            split=dl_args['subset'], mode=mode, target_type='semantic',
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
    elif dl_args["dataset"] == "cityscapes":
        return cityscapes.get_inr_loader_for_cityscapes(bsz=dl_args['batch size'],
            subset=dl_args['subset'], size=dl_args['image shape'])
    elif dl_args["dataset"] == "inet12":
        return inet.get_inr_loader_for_inet12(bsz=dl_args['batch size'], subset=dl_args['subset'])
    elif dl_args["dataset"] == "flowers":
        # return get_inr_loader_for_imgds('flowers', bsz=dl_args['batch size'], subset=dl_args['subset'])
        return get_inr_loader_for_flowers(bsz=dl_args['batch size'])
    else:
        raise NotImplementedError(dl_args["dataset"])

def get_inr_loader_for_cls_ds(ds_name, bsz, subset):
    inr = siren.Siren(out_channels=3)
    paths = glob2(f"{DS_DIR}/inrnet/{ds_name}/{subset}_*.pt")
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

def get_inr_loader_for_flowers(bsz):
    paths = glob2(f"{DS_DIR}/inrnet/flowers/*.pt")
    keys = siren.get_siren_keys()
    def random_loader():
        inrs = []
        while True:
            np.random.shuffle(paths)
            for path in paths:
                inr = siren.Siren(out_channels=3)
                param_dict = torch.load(path)
                inr.load_state_dict(param_dict)
                inrs.append(inr)
                if len(inrs) == bsz:
                    yield siren.to_black_box(inrs).cuda()
                    inrs = []
    return random_loader()


def get_inr_loader_for_imgds(ds_name, bsz, subset):
    paths = glob2(f"{DS_DIR}/inrnet/{ds_name}/{subset}_*.pt")
    if len(paths) == 0:
        raise ValueError('bad dataloader specs')
    keys = siren.get_siren_keys()
    def random_loader():
        inrs = []
        while True:
            np.random.shuffle(paths)
            for path in paths:
                inr = siren.Siren(out_channels=3)
                param_dict = torch.load(path)
                inr.load_state_dict(param_dict)
                inrs.append(inr)
                if len(inrs) == bsz:
                    yield siren.to_black_box(inrs).cuda()
                    inrs = []
    return random_loader()

