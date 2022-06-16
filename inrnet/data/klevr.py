import json
import os
osp=os.path
from inrnet.util import glob2

def get_klevr_paths():
    data_dir = '/data/vision/polina/scratch/clintonw/datasets/kubric-public/data'
    train_pattern = data_dir+'/multiview_matting/*/train/*'
    train_paths = glob2(train_pattern)
    val_pattern = data_dir+'/multiview_matting/*/val/*'
    val_paths = glob2(val_pattern)
    return train_paths, val_paths

def get_data():
    train_paths, val_paths = get_klevr_paths()
    for path in train_paths:
        data_json = osp.join(path, 'metadata.json')
        with open(data_json) as f:
            data = json.load(f)
        camera = data['camera']
        break