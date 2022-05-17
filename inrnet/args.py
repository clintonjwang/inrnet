import argparse, os, yaml, sys, shutil
osp = os.path

config_dir = osp.expanduser("~/code/diffcoord/configs")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name')
    parser.add_argument('-j', '--job_id', default="manual")
    parser.add_argument('-s', '--start_ix', default=0, type=int)
    parser.add_argument('-t', '--target_job', default=None)
    cmd_args = parser.parse_args()

    config_name = cmd_args.job_id if cmd_args.config_name is None else cmd_args.config_name
    main_config_path = osp.join(config_dir, config_name+".yaml")

    args = args_from_file(main_config_path, cmd_args)
    paths = args["paths"]
    for subdir in ("weights", "imgs", "plots"):
        shutil.rmtree(osp.join(paths["job output dir"], subdir), ignore_errors=True)
    for subdir in ("weights", "imgs", "plots"):
        os.makedirs(osp.join(paths["job output dir"], subdir))
    yaml.safe_dump(args, open(osp.join(paths["job output dir"], "config.yaml"), 'w'))
    return args

def infer_missing_args(args):
    paths = args["paths"]
    paths["slurm output dir"] = osp.expanduser(paths["slurm output dir"])
    if args["job_id"].startswith("lg_") or args["job_id"].startswith("A6"):
        args["data loading"]["batch size"] *= 6
        args["optimizer"]["epochs"] //= 6
    paths["job output dir"] = osp.join(paths["slurm output dir"], args["job_id"])
    paths["weights dir"] = osp.join(paths["job output dir"], "weights")
    for k in args["optimizer"]:
        if "learning rate" in k:
            args["optimizer"][k] = float(args["optimizer"][k])
    args["optimizer"]["weight decay"] = float(args["optimizer"]["weight decay"])

def merge_args(parent_args, child_args):
    if "_overwrite_" in child_args.keys():
        return child_args
    for k,parent_v in parent_args.items():
        if k not in child_args.keys():
            child_args[k] = parent_v
        else:
            if isinstance(child_args[k], dict) and isinstance(parent_v, dict):
                child_args[k] = merge_args(parent_v, child_args[k])
    return child_args


def args_from_file(path, cmd_args=None):
    config_dir = osp.dirname(path)
    if osp.exists(path):
        args = yaml.safe_load(open(path, 'r'))
    else:
        if cmd_args.config_name.endswith('1k'):
            path = path.replace('1k.yaml', '.yaml')
            if osp.exists(path):
                args = yaml.safe_load(open(path, 'r'))
                if 'data loading' in args: raise NotImplementedError
                args['data loading'] = {'N': 1000}
            else:
                raise ValueError(f"bad config_name {cmd_args.config_name}")
        elif cmd_args.config_name.endswith('4k'):
            path = path.replace('4k.yaml', '.yaml')
            if osp.exists(path):
                args = yaml.safe_load(open(path, 'r'))
                if 'data loading' in args: raise NotImplementedError
                args['data loading'] = {'N': 4000}
            else:
                raise ValueError(f"bad config_name {cmd_args.config_name}")
        else:
            raise ValueError(f"bad config_name {cmd_args.config_name}")

    if cmd_args is not None:
        for param in ["job_id", "config_name", "start_ix", 'target_job']:
            args[param] = getattr(cmd_args, param)

    while "parent" in args:
        if isinstance(args["parent"], str):
            config_path = osp.join(config_dir, args.pop("parent")+".yaml")
            args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
        else:
            parents = args.pop("parent")
            for p in parents:
                config_path = osp.join(config_dir, p+".yaml")
                args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
            if "parent" in args:
                raise NotImplementedError("need to handle case of multiple parents each with other parents")

    config_path = osp.join(config_dir, "default.yaml")
    args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
    infer_missing_args(args)
    return args
