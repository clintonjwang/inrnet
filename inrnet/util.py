import os, torch, cv2, math, pdb, torch, yaml, PIL
import numpy as np
import nibabel as nib
osp = os.path
from torch.nn.functional import avg_pool3d
import pandas as pd 
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import monai.transforms as mtr
from scipy.stats.qmc import Sobol

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_noclip = mtr.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)

def realign_values(out, coords_gt, inr=None, coords_out=None, split=None):
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
    if indices.size(0) != out.size(0):
        # print("realignment failed")
        # pdb.set_trace()
        raise ValueError("realignment failed")
    return out[indices]
    # O = coords_out.cpu().numpy().tolist()
    # indices = [O.index(G) for G in coords_gt.cpu().numpy()]
    # return out[indices]


def meshgrid_split_coords(*dims, split=2, domain=(-1,1), dtype=torch.half, device="cuda"):
    if len(dims) != 2 or split != 2:
        raise NotImplementedError

    tensors = [torch.linspace(*domain, steps=d) for d in dims]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    splitgrids = (mgrid[::2,::2], mgrid[1::2,::2], mgrid[::2,1::2], mgrid[1::2,1::2])
    return [mg.reshape(-1, len(dims)).to(dtype=dtype, device=device) for mg in splitgrids]

def meshgrid_coords(*dims, domain=(-1,1), dtype=torch.half, device="cuda"):
    tensors = [torch.linspace(*domain, steps=d) for d in dims]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid.reshape(-1, len(dims)).to(dtype=dtype, device=device)

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


def generate_quasirandom_sequence(d=3, n=64):
    sobol = Sobol(d=d)
    sample = sobol.random_base2(m=int(math.ceil(np.log2(n))))
    return sample

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

def draw_segs_as_contours(img, seg1, seg2, colors=([255, 0, 0], [0, 100, 255])):
    img, seg1, seg2 = [i.detach().cpu().squeeze().numpy() for i in (img, seg1, seg2)]
    img = to_rgb(img)
    contour1 = cv2.findContours((seg1>.5).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour1 = cv2.drawContours(np.zeros_like(img), contour1, -1, colors[0], 1)
    contour2 = cv2.findContours((seg2>.5).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour2 = cv2.drawContours(np.zeros_like(img), contour2, -1, colors[1], 1)
    # contour1[contour2 != 0] = 0
    img *= (contour1 == 0) * (contour2 == 0)
    img += contour1 + contour2
    return img

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









def get_per_patient_stats(img_names, metric_scores):
    '''
    Returns a dictionary with per-patient stats. Assumes the naming convention MAP-C303, etc...
    Inputs:
        img_names: list of image names
        metric_scores: list of corresponding metric scores
    Returns:
        dictionary with stats per patient
    '''
    name_subjs = [n[0:8] for n in img_names]
    setn = set(name_subjs)
    name_subjs = list(setn)
    metric_patient = dict()
    for name in name_subjs:
        inds = [ name in name_subjs for name_subjs in img_names]
        inds = [i for i, x in enumerate(inds) if x]
        metric_ct = 0.
        ct = 0
        for i in inds:
            metric_ct = metric_ct + metric_scores[i]
            ct = ct + 1
        metric_patient[name] = metric_ct/ct
    
    return metric_patient

def crop_range(img, shape, dim):
    """
    Computes the range of indices in the resized images for a given dimension. 

    Params:
    img: the original image
    shape: the resized shape
    dim: the dimension to compute

    Return: a list containing the range of indices in the resized image for a given dimension
    where the original image will be placed. 
    """
    offset = -(img.shape[dim] - shape[dim]) // 2
    if offset >= 0:
        return list(range(offset, offset + img.shape[dim]))
    else:
        return list(range(0, shape[dim]))

def get_crop_indices(img, shape):
    """
    Computes the range of indices in the resized images where the old image will be placed. 

    Params:
    img: the original image
    shape: the resized shape

    Return: a list containing the range of indices in the resized image
    where the original image will be placed. 
    """
    return crop_range(img, shape, 0), crop_range(img, shape, 1), crop_range(img, shape, 2)

def img_range(img, shape, dim):
    """
    Computes the range of indices in the original image that will be placed in the new image,
    for the given dimension. 

    Params: 
    img: the original image
    shape: the shape of the new image
    dim: the dimension to compute 

    Return: a list of indices. 
    """
    if img.shape[dim] <= shape[dim]:
        return list(range(0, img.shape[dim]))
    else:
        offset = (img.shape[dim] - shape[dim]) // 2
        return list(range(offset, offset + shape[dim]))

def get_img_indices(img, shape):
    """
    Computes the range of indices in the original image that will be placed in the new image.

    Params: 
    img: the original image
    shape: the shape of the new image

    Return: a list of indices. 
    """
    return img_range(img, shape, 0), img_range(img, shape, 1), img_range(img, shape, 2)

def crop_or_pad(img, shape, distr, labels=False):
    """
    crops or pads an image to the new shape, filling padding with noise. 

    Params:
    img: the original image
    shape: the desired size
    distr: the distribution for the padding noise
    labels: bool, True if the image is a label

    Return: the resized image
    """
    x, y, z = get_crop_indices(img, shape)
    new_img = np.zeros(shape)
    if not labels:
        new_img = np.random.normal(loc=distr[0], scale=distr[1], size=shape)
        new_img[new_img < 0] = 0
    x_crop, y_crop, z_crop = get_crop_indices(img, shape)
    x_img, y_img, z_img = get_img_indices(img, shape)
    new_img[np.ix_(x_crop, y_crop, z_crop)] = img[np.ix_(x_img, y_img, z_img)]
    return new_img

def load_img(path):
    """
    loads an image. 

    Params:
    path: the path to the file that should be loaded
    """
    img = nib.load(path)
    img = np.array(img.dataobj)
    return img.astype(float)

def save_img(data, path, fn):
    """
    saves an image as a nifti. 

    Params:
    data: the image to be saved. 
    path: the path to the file where the image should be saved. 
    fn: the filename that the image should be saved at. 
    """
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, osp.join(path, fn))

def find_patient_file(patient_id, data_dir, return_list=False): 
    """
    recursively looks for a filename including patient id in dir 
    
    patient_id: str patient id 
    data_dir: full directory path to search in
    return_list: if True, will return whole list of matches instead of just first
    """
    search = osp.join(data_dir, '**/{}*.nii*'.format(patient_id))
    files = glob(search, recursive = True) 
    if return_list:
        return files
    return files[0]

def best_metrics(csvfile, output_dir):
    """
    get best metrics from an existing csv file 
    """
    MAX_METRICS = ['val_dice','train_dice']
    MIN_METRICS = ['val_loss', 'train_loss']
    #get the rows with the best of each metric, save to csv
    df = pd.read_csv(csvfile)

    best_metrics = []
    for metric in MAX_METRICS:
        ind = df.index[df[metric] == df[metric].max()]
        s = df.iloc[ind,:]
        if len(s) > 1:
            s = s.iloc[0,:]
        best_metrics.append(s)
    for metric in MIN_METRICS:
        ind = df.index[df[metric] == df[metric].min()]
        s = df.iloc[ind,:]        
        if len(s) > 1:
            s = s.iloc[0,:]
        best_metrics.append(s)
    best_of_each_df = pd.concat(best_metrics)
    best_of_each_df["optimized_metric"] = MAX_METRICS+MIN_METRICS
    best_of_each_df.to_csv(osp.join(output_dir,'best_metrics.csv'), index=False)
