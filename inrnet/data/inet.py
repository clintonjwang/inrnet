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

def get_imagenet_classes():
    classes = open(DS_DIR+"/imagenet_pytorch/labels.txt", 'r').read().split('\n')
    return [c[:c.find(',')] for c in classes]

class INet12(torchvision.datasets.VisionDataset):
    def __init__(self, subset, cls, *args, **kwargs):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
                classes.append(torch.tensor(sub_to_super[cl]).cuda())

                if len(inrs) == bsz:
                    yield inrs, torch.stack(classes)
                    inrs = []
                    classes = []
    return random_loader()

