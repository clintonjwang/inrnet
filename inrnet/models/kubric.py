import os, glob, json
import shutil
osp = os.path

def clean_kubric():
    """
    Clean the kubric directory.
    """
    subdirs = glob.glob(osp.expanduser("~/code/kubric/output/*"))
    for subdir in subdirs:
        if "rgba_00002.png" not in os.listdir(subdir):
            shutil.rmtree(subdir)

def tmp():
    subdirs = glob.glob(osp.expanduser("~/code/kubric/output/*"))
    for subdir in subdirs:
        path = subdir+"/transforms_test.json"
        if osp.exists(path):
            os.remove(path)
    # src = osp.expanduser(f"~/code/kubric/output/39/transforms_test.json")
    # for ix in range(39):
    #     target = osp.expanduser(f"~/code/kubric/output/{ix}/transforms_test.json")
    #     shutil.copy(src, target)

#import inrnet.models.kubric as K
#K.clean_kubric()
#K.tmp()