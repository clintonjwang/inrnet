import os, torch, pdb, pickle
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from inrnet.models.inrs import siren
from inrnet.util import glob2

DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

def get_imagenet_classes():
    classes = open(DS_DIR+"/imagenet_pytorch/labels.txt", 'r').read().split('\n')
    return [c[:c.find(',')] for c in classes]

class INet12(torchvision.datasets.VisionDataset):
    def __init__(self, subset, cls, *args, **kwargs):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        root = DS_DIR+"/imagenet_pytorch"
        super().__init__(root, *args, transform=trans, **kwargs)
        big12path = f"{DS_DIR}/inrnet/big_12_labels.pkl"
        split_paths_by_cls, _, _ = pickle.load(open(big12path, 'rb'))
        self.subpaths = split_paths_by_cls[subset][cls]

    def __getitem__(self, ix):
        img = Image.open(osp.join(self.root, self.subpaths[ix]))
        try:
            return self.transform(img)
        except RuntimeError:
            return self.transform(img.convert('RGB'))
    def __len__(self):
        return len(self.subpaths)


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

def get_inr_loader_for_inet12(bsz, subset):
    paths = [glob2(f"{DS_DIR}/inrnet/inet12/{c}/{subset}_*.pt") for c in range(12)]
    if subset == 'train':
        N = 800
        loop = True
    elif subset == 'test':
        N = 200
        loop = False
    else:
        raise NotImplementedError(f'bad subset {subset}')
    for p in paths:
        assert len(p)==N, f'incomplete subset ({len(p)}/{N})'

    keys = siren.get_siren_keys()
    if bsz % 12 == 0:
        n_per_cls = bsz // 12
        if N % bsz != 0:
            print('warning: dropping last minibatch')
            N = (N // bsz) * bsz
        while True:
            for p in paths:
                np.random.shuffle(p)
            for path_ix in range(0, N, n_per_cls):
                inrs = [siren.Siren(out_channels=3) for _ in range(bsz)]
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
                yield siren.to_black_box(inrs).cuda(), labels.cuda()
            if not loop:
                return

    elif bsz == 1:
        print('analysis mode only')
        for path_ix in range(N):
            inr = siren.Siren(out_channels=3)
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
    


