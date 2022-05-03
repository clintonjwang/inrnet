import os, torch, pdb, pickle
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
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
    def __init__(self, root, subset, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.subpaths = open(self.root+f"/{subset}.txt", "r").read().split('\n')
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
        dataset = INetDS(DS_DIR+"/imagenet_pytorch", transform=trans, subset=dl_args['subset'])

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
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = torchvision.datasets.ImageFolder(
            DS_DIR+"/places365_standard/val", transform=trans)

    elif dl_args["dataset"] == "cifar10":
        return torchvision.datasets.CIFAR10(root=DS_DIR, train=dl_args['subset'] == 'train',
            transform=transforms.ToTensor())

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
        return get_inr_loader_for_inet12(bsz=dl_args['batch size'], subset=dl_args['subset'])
    else:
        raise NotImplementedError(dl_args["dataset"])

def get_imagenet_classes():
    classes = open(DS_DIR+"/imagenet_pytorch/labels.txt", 'r').read().split('\n')
    return [c[:c.find(',')] for c in classes]

def get_inr_loader_for_inet12(bsz, subset):
    cls_map_path = f"{DS_DIR}/inrnet/big_12.pkl"
    _, _, sub_to_super = pickle.load(open(cls_map_path, 'rb'))

    inr = siren.Siren(out_channels=3)
    paths = glob2(f"{DS_DIR}/inrnet/imagenet1k_{subset}/siren_*.pt")
    keys = siren.get_siren_keys()
    def random_loader():
        inrs = []
        classes = []
        while True:
            np.random.shuffle(paths)
            for path in paths:
                data, cl = torch.load(path)
                indices = list(range(len(data)))
                # for ix in indices:
                ix = np.random.choice(indices)
                param_dict = {k:data[k][ix] for k in keys}
                try:
                    inr.load_state_dict(param_dict)
                except RuntimeError:
                    continue
                cl = cl[ix]['cls']
                if cl not in sub_to_super:
                    continue
                inrs.append(inr.cuda())
                classes.append(torch.tensor(sub_to_super[cl]).unsqueeze(0).cuda())

                if len(inrs) == bsz:
                    yield inrs, torch.stack(classes)
    return random_loader()


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
                # for ix in indices:
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

