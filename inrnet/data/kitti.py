# adapted from https://github.com/nianticlabs/monodepth2
import os, torch
osp = os.path
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms

from inrnet.models.inrs import siren
from inrnet import util

kitti_root = "/data/vision/torralba/datasets/kitti_raw"
DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"
TMP_DIR = osp.expanduser("~/code/diffcoord/temp")

def get_kitti_inr_dataloader():
    grayscale = siren.Siren(out_channels=1)
    rgb = siren.Siren(out_channels=3)
    paths = sorted(util.glob2(f"{DS_DIR}/inrnet/kitti/siren_*.pt"))
    keys = ['net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight', 'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight', 'net.3.linear.bias', 'net.4.weight', 'net.4.bias']
    for path in paths:
        data = torch.load(path)
        for ix in range(len(data[1])):
            param_dict = {k:data[0][k][ix] for k in keys}
            depth = data[1][ix]["depth"]
            if param_dict["net.4.bias"].size(0) == 3:
                rgb.load_state_dict(param_dict)
                yield rgb.cuda(), depth.cuda()
            else:
                grayscale.load_state_dict(param_dict)
                yield grayscale.cuda(), depth.cuda()

    
def filter_kitti_imgs():
    from PIL import Image
    from torchvision import transforms
    to_torch = transforms.ToTensor()
    paths = sorted(util.glob2("~/code/diffcoord/temp/kitti/depth_*.bin"))
    dps = []
    ix = 0
    for path in paths:
        data = pickle.load(open(path, "rb"))
        for dp in data:
            img = to_torch(Image.open(dp["img_path"]))
            if img.size(0) == 3:
                dps.append(dp)
                if len(dps) == 64:
                    pickle.dump(dps, open(osp.expanduser(f"~/code/diffcoord/temp/kitti/rgb_{ix}.bin"), "wb"))
                    ix+=64
                    dps=[]

def get_kitti_img_dataloader():
    to_torch = transforms.ToTensor()
    paths = sorted(util.glob2("~/code/diffcoord/temp/kitti/rgb_*.bin"))
    for path in paths:
        data = pickle.load(open(path, "rb"))
        for dp in data:
            img = to_torch(Image.open(dp["img_path"]))
            if len(img.shape) == 3:
                img.unsqueeze_(0)
            yield {"img":img, "id":dp["id"], "depth":to_torch(dp["depth"])}

def save_kitti_imgs(start_ix=0):
    start_ix=int(start_ix)
    date_paths = util.glob2(kitti_root, "201*")
    datapoints = []
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    ix = 0
    for date_path in date_paths:
        drive_paths = util.glob2(date_path)
        calib_dir = date_path
        for drive_path in drive_paths:
            map_paths = util.glob2(drive_path, "image_0*/data")
            for map_path in map_paths:
                side = map_path[-6]
                frame_paths = util.glob2(map_path, "*.png")
                for frame_path in frame_paths:
                    if ix < start_ix:
                        ix += 1
                        continue
                    frame_index = osp.basename(frame_path)[:-4]
                    try:
                        velo_filename = osp.join(f"{drive_path}/velodyne_points/data/{frame_index}.bin")
                        xyz = generate_depth_map(calib_dir, velo_filename, cam=side_map[side] if side in side_map else 2)
                    except FileNotFoundError:
                        if ix % 64 == 63:
                            pickle.dump(datapoints, open(f"{TMP_DIR}/kitti/depth_{ix}.bin", "wb"))
                            datapoints = []
                        ix += 1
                        continue
                        
                    img_id = osp.basename(drive_path)+f"-{side}-{frame_index[-4:]}"
                    datapoints.append({"id":img_id, "img_path":frame_path, "depth":xyz})

                    if ix % 64 == 63:
                        pickle.dump(datapoints, open(f"{TMP_DIR}/kitti/depth_{ix}.bin", "wb"))
                        datapoints = []

                    ix += 1


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def read_calib_file(path):
    """Read KITTI calibration file
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def generate_depth_map(calib_dir, velo_filename, cam=2):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(osp.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(osp.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]
    return velo_pts_im
