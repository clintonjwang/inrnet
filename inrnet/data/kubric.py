import json
from inrnet.utils.util import glob2
import os, glob, json
import shutil
osp = os.path

def clean_kubric():
    """
    Clean the kubric directory.
    """
    subdirs = glob.glob(osp.expanduser("~/code/kubric/output/*"))
    for subdir in subdirs:
        if osp.isdir(subdir) and "rgba_00002.png" not in os.listdir(subdir):
            shutil.rmtree(subdir)

def tmp():
    subdirs = glob.glob(osp.expanduser("~/code/kubric/output/*"))
    for subdir in subdirs:
        if osp.isdir(subdir):
            path = subdir+"/transforms.json"
            if osp.exists(path):
                os.rename(path, path.replace("transforms.json",
                    "transforms_train.json"))
    # src = osp.expanduser(f"~/code/kubric/output/39/transforms_test.json")
    # for ix in range(39):
    #     target = osp.expanduser(f"~/code/kubric/output/{ix}/transforms_test.json")
    #     shutil.copy(src, target)

#import inrnet.models.kubric as K
#K.clean_kubric()
#K.tmp()
# def get_kubric_paths():
#     data_dir = '/data/vision/polina/scratch/clintonw/datasets/kubric-public/data'
#     train_pattern = data_dir+'/multiview_matting/*/train/*'
#     train_paths = glob2(train_pattern)
#     val_pattern = data_dir+'/multiview_matting/*/val/*'
#     val_paths = glob2(val_pattern)
#     return train_paths, val_paths

# def get_data():
#     train_paths, val_paths = get_kubric_paths()
#     for path in train_paths:
#         data_json = osp.join(path, 'metadata.json')
#         with open(data_json) as f:
#             data = json.load(f)
#         camera = data['camera']
#         break