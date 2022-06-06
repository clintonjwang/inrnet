import os
import torch
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
nn=torch.nn
from inrnet.models.inrs import siren
from inrnet.util import glob2
from inrnet.data import kitti, inet, cityscapes
nearest = transforms.InterpolationMode('nearest')

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
        kitti.get_kitti_img_dataloader()

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

    elif dl_args["dataset"] == "fmnist":
        return torchvision.datasets.FashionMNIST(root=DS_DIR, train=dl_args['subset'] == 'train',
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
            subset=dl_args['subset'], size=dl_args['image shape'], mode=dl_args['seg type'])
    elif dl_args["dataset"] == "inet12":
        return inet.get_inr_loader_for_inet12(bsz=dl_args['batch size'], subset=dl_args['subset'],
            N=dl_args['N'])
    elif dl_args["dataset"] == "fmnist":
        return get_inr_loader_for_fmnist(bsz=dl_args['batch size'])
    elif dl_args["dataset"] == "oasis":
        return get_inr_loader_for_oasis(bsz=dl_args['batch size'],
            size=dl_args['image shape'], subset=dl_args['subset'])
    elif dl_args["dataset"] == "flowers":
        # return get_inr_loader_for_imgds('flowers', bsz=dl_args['batch size'], subset=dl_args['subset'])
        return get_inr_loader_for_flowers(bsz=dl_args['batch size'])
    else:
        raise NotImplementedError(dl_args["dataset"])

def get_val_inr_dataloader(dl_args):
    dl_args['subset'] = 'val'
    if dl_args["dataset"] == "inet12":
        dl_args['batch size'] = 192
    else:
        dl_args['batch size'] *= 4
    while True:
        dl = get_inr_dataloader(dl_args)
        for data in dl:
            yield data

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

def get_inr_loader_for_fmnist(bsz, N=5000, subset='train'):
    ix_to_cls = torch.load(f"{DS_DIR}/inrnet/fmnist/ix_to_cls.pt")
    paths = [f"{DS_DIR}/inrnet/fmnist/{subset}_{ix}.pt" for ix in range(60000) if ix_to_cls[ix] in [5,7,9]]
    paths = [p for p in paths if osp.exists(p)][:N]
    def random_loader():
        inrs = []
        while True:
            np.random.shuffle(paths)
            for path in paths:
                inr = siren.Siren(out_channels=1, C=32, first_omega_0=30, hidden_omega_0=30)
                param_dict,_ = torch.load(path)
                inr.load_state_dict(param_dict)
                inrs.append(inr)
                if len(inrs) == bsz:
                    yield siren.to_black_box(inrs).cuda()
                    inrs = []
    return random_loader()

def get_inr_loader_for_flowers(bsz):
    paths = glob2(f"{DS_DIR}/inrnet/flowers/*.pt")
    siren.get_siren_keys()
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


def get_inr_loader_for_oasis(bsz, size, subset):
    paths = glob2(f"{DS_DIR}/inrnet/oasis/{subset}_*_0.pt")
    shrink = transforms.Resize(size, interpolation=nearest)
    loop = subset == 'train'
    N = len(paths)
    N = (N // bsz) * bsz
    def random_loader(loop=True):
        while True:
            np.random.shuffle(paths)
            for path_ix in range(0,N,bsz):
                start = np.random.randint(3)
                s_path = paths[path_ix].replace('0.pt', f'{start}.pt')
                t_path = paths[path_ix].replace('0.pt', f'{start+1}.pt')

                inrs = [siren.Siren(out_channels=1) for _ in range(bsz*2)]
                merged_inrs = []
                segs = []
                for i in range(bsz):
                    param_dict, sseg = torch.load(s_path)
                    inrs[i*2].load_state_dict(param_dict)
                    param_dict, tseg = torch.load(t_path)
                    inrs[i*2+1].load_state_dict(param_dict)
                    segs.append(torch.cat([sseg, tseg], dim=1))
                    merged_inrs.append(MergeSIREN(inrs[i*2:i*2+2]))
                segs = torch.cat(segs, dim=0).cuda()
                yield siren.to_black_box(merged_inrs).cuda(), shrink(segs)
            if not loop:
                break
    
    return random_loader(loop=loop)

class MergeSIREN(nn.Module):
    def __init__(self, inrs):
        super().__init__()
        self.evaluator = nn.ModuleList(inrs).eval()
        self.out_channels = len(inrs) * inrs[0].out_channels
    def forward(self, coords):
        out = []
        for inr in self.evaluator:
            out.append(inr(coords))
        return torch.cat(out, dim=0)


def get_inr_loader_for_imgds(ds_name, bsz, subset):
    paths = glob2(f"{DS_DIR}/inrnet/{ds_name}/{subset}_*.pt")
    if len(paths) == 0:
        raise ValueError('bad dataloader specs')
    siren.get_siren_keys()
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

