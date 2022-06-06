import os
import pickle
import torch
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from inrnet.inn.transforms import coord_noise, rand_flip, intensity_noise
from inrnet.models.inrs import siren, rff

DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

def get_imagenet_classes():
    classes = open(DS_DIR+"/imagenet_pytorch/labels.txt", 'r').read().split('\n')
    return [c[:c.find(',')] for c in classes]

class INet12(torchvision.datasets.VisionDataset):
    def __init__(self, subset, *args, **kwargs):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        root = DS_DIR+"/imagenet_pytorch"
        super().__init__(root, *args, transform=trans, **kwargs)
        big12path = f"{DS_DIR}/inrnet/big_12_labels.pkl"
        split_paths_by_cls, _, _ = pickle.load(open(big12path, 'rb'))
        self.subpaths = split_paths_by_cls[subset]

    def __getitem__(self, ix):
        cls, ix = ix % 12, ix // 12
        img = Image.open(osp.join(self.root, self.subpaths[cls][ix]))
        try:
            return self.transform(img)
        except RuntimeError:
            return self.transform(img.convert('RGB'))
    def __len__(self):
        return len(self.subpaths)


class INet12Extra(torchvision.datasets.VisionDataset):
    def __init__(self, *args, **kwargs):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
        ])
        root = DS_DIR+"/imagenet_pytorch"
        super().__init__(root, *args, transform=trans, **kwargs)
        big12path = f"{DS_DIR}/inrnet/big_12_extra.pkl"
        self.extra_subpaths = pickle.load(open(big12path, 'rb'))

    def __getitem__(self, ix):
        cls, ix = ix % 12, ix // 12
        img = Image.open(osp.join(self.root, self.extra_subpaths[cls][ix]))
        try:
            return self.transform(img)
        except RuntimeError:
            return self.transform(img.convert('RGB'))
    def __len__(self):
        return len(self.extra_subpaths[0])*12



class INetDS(torchvision.datasets.VisionDataset):
    def __init__(self, root, subset, *args, **kwargs):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        super().__init__(root, *args, transform=trans, **kwargs)
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

def get_inr_loader_for_inet12(bsz, subset, N=None):
    if subset == 'train':
        if N is None:
            N = 500
        paths = [[f"{DS_DIR}/inrnet/inet12/{c}/{subset}_{ix}.pt" for ix in range(N)] \
            for c in range(12)]
        loop = True

        spatial_augs = [coord_noise, rand_flip]
        intensity_augs = [intensity_noise]
    elif subset in ('test', 'val'):
        N = 200
        paths = [[f"{DS_DIR}/inrnet/inet12/{c}/{subset}_{ix}.pt" for ix in range(N)] \
            for c in range(12)]
        loop = False
    else:
        raise NotImplementedError(f'bad subset {subset}')
    # for p in paths:
    #     assert len(p)==N, f'incomplete subset ({len(p)}/{N})'

    # def aug_coords(values):
    #     return (values - torch.tensor((0.485, 0.456, 0.406), device=values.device)) / torch.tensor(
    #         (0.229, 0.224, 0.225), device=values.device)
    if subset in ('train', 'val'): #SIREN
        model = siren.Siren
        inet_rescale = None
    else:
        def inet_rescale(values):
            return (values - torch.tensor((.5), device=values.device)) / torch.tensor(
                (.5), device=values.device)
        model = rff.RFFNet

    if bsz % 12 == 0:
        n_per_cls = bsz // 12
        if N % bsz != 0:
            print('warning: dropping last minibatch')
            if bsz > N:
                raise NotImplementedError
            N = (N // bsz) * bsz
        while True:
            for p in paths:
                np.random.shuffle(p)
            for path_ix in range(0, N, n_per_cls):
                inrs = [model() for _ in range(bsz)]
                for i in range(n_per_cls):
                    for cl in range(12):
                        param_dict = torch.load(paths[cl][path_ix+i])
                        try:
                            inrs[i*12+cl].load_state_dict(param_dict)
                        except RuntimeError:
                            param_dict['net.4.weight'] = param_dict['net.4.weight'].tile(3,1)
                            param_dict['net.4.bias'] = param_dict['net.4.bias'].tile(3)
                            inrs[i*12+cl].load_state_dict(param_dict)
                labels = torch.arange(12).tile(n_per_cls)
                if model == rff.RFFNet:
                    inrs = rff.to_black_box(inrs).cuda()
                else:
                    inrs = siren.to_black_box(inrs).cuda()
                if inet_rescale is not None:
                    inrs.add_transforms(intensity=inet_rescale)
                if subset == 'train':
                    inrs.add_transforms(spatial=spatial_augs, intensity=intensity_augs)
                yield inrs, labels.cuda()
            if not loop:
                return

    elif bsz == 1:
        print('analysis mode only')
        for path_ix in range(N):
            inr = model()
            for cl in range(12):
                param_dict = torch.load(paths[cl][path_ix])
                try:
                    inr.load_state_dict(param_dict)
                except RuntimeError:
                    param_dict['net.4.weight'] = param_dict['net.4.weight'].tile(3,1)
                    param_dict['net.4.bias'] = param_dict['net.4.bias'].tile(3)
                    inr.load_state_dict(param_dict)
                yield siren.to_black_box([inr]).cuda(), torch.tensor([cl]).cuda()
    else:
        raise ValueError('expect batch size to be a multiple of the number of classes')



def analyze_inr_error():
    INet12('train')
    return

# def generate_inet12(n_train=800, n_val=200):
#     cls_map_path = f"{DS_DIR}/inrnet/big_12.pkl"
#     _, _, sub_to_super = pickle.load(open(cls_map_path, 'rb'))
#     N = n_train + n_val
#     keys = siren.get_siren_keys()

#     paths = glob2(f"{DS_DIR}/inrnet/imagenet1k_*/siren_*.pt")
#     data_by_cls = [[] for _ in range(12)]
#     np.random.shuffle(paths)
#     for path in paths:
#         data, C = torch.load(path)
#         indices = list(range(len(data)))
#         np.random.shuffle(indices)
#         for ix in indices:
#             cl = C[ix]['cls']
#             if cl not in sub_to_super:
#                 continue
#             cl = sub_to_super[cl]
#             if data_by_cls[cl] is None:
#                 continue
#             data_by_cls[cl].append([data[k][ix] for k in keys])
#             if len(data_by_cls[cl]) == N:
#                 print(f'done with class {cl}')
#                 np.random.shuffle(data_by_cls[cl])
#                 torch.save(data_by_cls[cl][:n_train], f"{DS_DIR}/inrnet/imagenet1k_val/{cl}_train.pt")
#                 torch.save(data_by_cls[cl][n_train:], f"{DS_DIR}/inrnet/imagenet1k_val/{cl}_test.pt")
#                 data_by_cls[cl] = None
#                 break
    
