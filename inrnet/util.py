import os, torch, math, pdb, yaml, PIL
import numpy as np
osp = os.path
from glob import glob
import matplotlib.pyplot as plt
import monai.transforms as mtr
from scipy.stats.qmc import Sobol
import seaborn as sns

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_noclip = mtr.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)

def realign_values(out, coords_gt, inr=None, coords_out=None, split=None):
    # out - (B,N,C)
    # coords_gt - (N,D)

    if coords_out is None:
        coords_out = inr.sampled_coords
    # coords_gt = coords_gt[:,0]*4 + coords_gt[:,1]
    # coords_out = coords_out[:,0]*4 + coords_out[:,1]
    if split is None:
        # matches = (coords_gt.unsqueeze(1) == coords_out.unsqueeze(0)).min(dim=-1).values
        diffs = coords_out.unsqueeze(0) - coords_gt.unsqueeze(1)
        matches = diffs.abs().sum(-1) == 0
        indices = torch.where(matches)[1]
    else:
        N = coords_out.size(0)
        dx = N//split
        indices = []
        for ix in range(0,N,dx):
            diffs = coords_out.unsqueeze(0) - coords_gt[ix:ix+dx].unsqueeze(1)
            matches = diffs.abs().sum(-1) == 0
            indices.append(torch.where(matches)[1])
            # matches = (coords_gt[ix:ix+dx].unsqueeze(1) == coords_out.unsqueeze(0)).min(dim=-1).values
            # indices.append(torch.where(matches)[1])
            del diffs, matches
            torch.cuda.empty_cache()
        indices = torch.cat(indices,dim=0)
    if indices.size(0) != out.size(1):
        print("realignment failed")
        pdb.set_trace()
        # raise ValueError("realignment failed")
    return out[:,indices]

def meshgrid(*tensors, indexing='ij'):
    try:
        return torch.meshgrid(*tensors, indexing=indexing)
    except TypeError:
        return torch.meshgrid(*tensors)

def meshgrid_coords(*dims, domain=(-1,1), dtype=torch.float, device="cuda"):
    tensors = [torch.linspace(*domain, steps=d) for d in dims]
    mgrid = torch.stack(meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid.reshape(-1, len(dims)).to(dtype=dtype, device=device)


def meshgrid_split_coords(*dims, split=2, domain=(-1,1), dtype=torch.float, device="cuda"):
    if len(dims) != 2 or split != 2:
        raise NotImplementedError

    tensors = [torch.linspace(*domain, steps=d) for d in dims]
    mgrid = torch.stack(meshgrid(*tensors, indexing='ij'), dim=-1)
    splitgrids = (mgrid[::2,::2], mgrid[1::2,::2], mgrid[::2,1::2], mgrid[1::2,1::2])
    return [mg.reshape(-1, len(dims)).to(dtype=dtype, device=device) for mg in splitgrids]

def first_split_meshgrid(*dims, split=2, domain=(-1,1), dtype=torch.float, device="cuda"):
    tensors = [torch.linspace(*domain, steps=d) for d in dims]
    mgrid = torch.stack(meshgrid(*tensors, indexing='ij'), dim=-1)
    splitgrid = mgrid[::split,::split]
    return splitgrid.reshape(-1, len(dims)).to(dtype=dtype, device=device)

def load_checkpoint(model, paths):
    if paths["pretrained model name"] is not None:
        init_weight_path = osp.join(TMP_DIR, paths["pretrained model name"])
        if not osp.exists(init_weight_path):
            raise ValueError(f"bad pretrained model path {init_weight_path}")

        checkpoint_sd = torch.load(init_weight_path)
        model_sd = model.state_dict()
        for k in model_sd.keys():
            if k in checkpoint_sd.keys() and checkpoint_sd[k].shape != model_sd[k].shape:
                checkpoint_sd.pop(k)

        model.load_state_dict(checkpoint_sd, strict=False)


def generate_quasirandom_sequence(d=3, n=64, like=None):
    sobol = Sobol(d=d)
    sample = sobol.random_base2(m=int(math.ceil(np.log2(n))))
    if like is None:
        return sample
    else:
        return torch.tensor(sample, dtype=like.dtype, device=like.device)

def cycle(iterator):
    while True:
        for data in iterator:
            yield data

def parse_int_or_list(x):
    # converts string to an int or list of ints
    if not isinstance(x, str):
        return x
    try:
        return int(x)
    except ValueError:
        return [int(s.strip()) for s in x.split(',')]

def parse_float_or_list(x):
    # converts string to a float or list of floats
    if not isinstance(x, str):
        return x
    try:
        return float(x)
    except ValueError:
        return [float(s.strip()) for s in x.split(',')]

def glob2(*paths):
    pattern = osp.expanduser(osp.join(*paths))
    if "*" not in pattern:
        pattern = osp.join(pattern, "*")
    return glob(pattern)

def flatten_list(collection):
    new_list = []
    for element in collection:
        new_list += list(element)
    return new_list


class MetricTracker:
    def __init__(self, name=None, intervals=None, function=None, weight=1.):
        self.name = name
        self.epoch_history = {"train":[], "val":[]}
        if intervals is None:
            intervals = {"train":1,"val":1}
        self.intervals = intervals
        self.function = function
        self.minibatch_values = {"train":[], "val":[]}
        self.weight = weight
    
    def __call__(self, *args, phase="val", **kwargs):
        loss = self.function(*args, **kwargs)
        self.update_with_minibatch(loss, phase=phase)
        # if np.isnan(loss.mean().item()):
        #     raise ValueError(f"{self.name} became NaN")
        return loss.mean() * self.weight

    def update_at_epoch_end(self):
        for phase in self.minibatch_values:
            if len(self.minibatch_values[phase]) != 0:
                self.epoch_history[phase].append(self.epoch_average(phase))
                self.minibatch_values[phase] = []

    def update_with_minibatch(self, value, phase="val"):
        if isinstance(value, torch.Tensor):
            if torch.numel(value) == 1:
                self.minibatch_values[phase].append(value.item())
            else:
                self.minibatch_values[phase] += list(value.detach().cpu().numpy())
        elif isinstance(value, list):
            self.minibatch_values[phase] += value
        elif isinstance(value, np.ndarray):
            self.minibatch_values[phase] += list(value)
        elif not np.isnan(value):
            self.minibatch_values[phase].append(value)

    def epoch_average(self, phase="val"):
        if len(self.minibatch_values[phase]) != 0:
            return np.nanmean(self.minibatch_values[phase])
        elif len(self.epoch_history[phase]) != 0:
            return self.epoch_history[phase][-1]
        return np.nan

    def max(self, phase="val"):
        try: return np.nanmax(self.epoch_history[phase])
        except: return np.nan
    def min(self, phase="val"):
        try: return np.nanmin(self.epoch_history[phase])
        except: return np.nan

    def current_moving_average(self, window, phase="val"):
        return np.mean(self.minibatch_values[phase][-window:])

    def get_moving_average(self, window, interval=None, phase="val"):
        if interval is None:
            interval = window
        ret = np.cumsum(self.minibatch_values[phase], dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        ret = ret[window - 1:] / window
        return ret[::interval]

    def is_at_max(self, phase="val"):
        return self.max(phase) >= self.epoch_average(phase)
    def is_at_min(self, phase="val"):
        return self.min(phase) <= self.epoch_average(phase)

    def histogram(self, path=None, phase="val", epoch=None):
        if epoch is None:
            epoch = len(self.epoch_history[phase]) * self.intervals[phase]
        if len(self.minibatch_values[phase]) < 5:
            return
        plt.hist(self.minibatch_values[phase])
        plt.title(f"{self.name}_{phase}_{epoch}")
        if path is not None:
            plt.savefig(path)
            plt.clf()

    def plot_running_average(self, path, phase='val'):
        _,axis = plt.subplots()
        N = 10
        x = self.epoch_history[phase]
        if len(x) == 0:
            x = self.minibatch_values[phase]
        if len(x) <= N:
            return
        x = np.convolve(x, np.ones(N)/N, mode='valid')
        sns.lineplot(y=x[::N//2], ax=axis, label=phase)
        axis.set_ylabel(self.name)
        plt.savefig(path)
        plt.close()

    def lineplot(self, path=None):
        _,axis = plt.subplots()

        if "train" in self.intervals:
            dx = self.intervals["train"]
            values = self.epoch_history["train"]
            if len(values) >= 3:
                x_values = np.arange(0, dx*len(values), dx)
                sns.lineplot(x=x_values, y=values, ax=axis, label="train")
        
        if "val" in self.intervals:
            dx = self.intervals["val"]
            values = self.epoch_history["val"]
            if len(values) >= 3:
                x_values = np.arange(0, dx*len(values), dx)
                sns.lineplot(x=x_values, y=values, ax=axis, label="val")

        axis.set_ylabel(self.name)

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            return axis

def save_plots(trackers, root=None):
    os.makedirs(root, exist_ok=True)
    for tracker in trackers:
        path = osp.join(root, tracker.name+".png")
        tracker.lineplot(path=path)


def save_metric_histograms(trackers, epoch, root):
    os.makedirs(root, exist_ok=True)
    for tracker in trackers:
        for phase in ["train", "val"]:
            path = osp.join(root, f"{epoch}_{tracker.name}_{phase}.png")
            tracker.histogram(path=path, phase=phase, epoch=epoch)


def save_examples(prefix, root, *imgs):
    imgs = list(imgs)
    if isinstance(imgs[0], torch.Tensor):
        for ix in range(len(imgs)):
            imgs[ix] = imgs[ix].detach().cpu().squeeze(1).numpy()
        if transforms is not None:
            transforms = transforms.detach().cpu().numpy()

    os.makedirs(root, exist_ok=True)
    for ix in range(imgs[0].shape[0]):
        cat = np.concatenate([img[ix] for img in imgs], axis=1)
        cat = rescale_clip(cat)
        plt.imsave(f"{root}/{prefix}_{ix}.png", cat, cmap="gray")
    plt.close("all")

def to_rgb(img):
    if isinstance(img, torch.Tensor):
        if len(img.shape)!=2:
            raise NotImplementedError
        img = img.tile(3,1,1)
    else:
        img = np.stack([img, img, img], -1)
    return img

def save_example(path, img, gt_seg, pred_logit):
    img, gt_seg, pred_logit = [i.squeeze().detach().cpu() for i in (img, gt_seg, pred_logit)]
    x = gt_seg.sum((1,2)).argmax().item()
    y = gt_seg.sum((0,2)).argmax().item()
    z = gt_seg.sum((0,1)).argmax().item()
    img = rescale_clip(img.float())
    pred_seg = pred_logit > 0
    pred_logit = rescale_noclip(pred_logit.float())
    w = max(img.shape[-2:])
    pad = mtr.SpatialPad((-1,w))
    padd = lambda q: pad(q.unsqueeze(0)).squeeze(0)
    I,G,PL,PS = [(padd(i[x]), padd(i[:,y]), padd(i[:,:,z])) for i in (img, gt_seg, pred_logit, pred_seg)]
    overlays = [torch.tensor(arr) for arr in (
        draw_segs_as_contours(I[0], G[0], PS[0]),
        draw_segs_as_contours(I[1], G[1], PS[1]),
        draw_segs_as_contours(I[2], G[2], PS[2]))]
    col1 = torch.cat(overlays, dim=0)
    col2 = to_rgb(torch.cat(I, dim=0)).permute(1,2,0)
    col3 = to_rgb(torch.cat(PL, dim=0)).permute(1,2,0)
    img = torch.cat((col1, col2, col3), dim=1)

    os.makedirs(osp.dirname(path), exist_ok=True)
    try:
        plt.imsave(path, img.numpy())
        plt.close()
    except ValueError:
        pass #encountered empty seg

def save_example_slices(img, gt_seg, pred_seg, root):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze().numpy()
        gt_seg = gt_seg.detach().cpu().squeeze().numpy()
        pred_seg = pred_seg.detach().cpu().squeeze().numpy()

    rescale = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
    # rescale = mtr.ScaleIntensity(minv=0, maxv=255, dtype=np.uint8)
    img = rescale(img)

    os.makedirs(root, exist_ok=True)

    for z in range(5,76,5):
        img_slice = img[:,:,z]
        gt_seg_slice = gt_seg[:,:,z].round()
        pred_seg_slice = pred_seg[:,:,z] > 0
        plt.imsave(f"{root}/gt_{z}.png", gt_seg_slice)
        plt.imsave(f"{root}/pred_{z}.png", pred_seg_slice)

        img_slice = draw_segs_as_contours(img_slice, gt_seg_slice, pred_seg_slice)
        plt.imsave(f"{root}/{z}.png", img_slice)
    plt.close("all")

