from __future__ import print_function
import torch
import torchvision
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import random
import logging

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
# Li Jiaxin, because of the tanh() of the generator, here this function assumes that the output is in [-1,1]
# but actually for depth estimation, this function is used only for input rgb, hence is the inverse of ([0-1]-0.5)/0.5
# in this function, CxHxW -> HxWxC
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1,2,0)) * 0.5 + 0.5) * 255.0
    return image_numpy.astype(imtype)

# Li Jiaxin, for test images
def tensor2grid_im(image_tensor):
    grid = torchvision.utils.make_grid(image_tensor, nrow=5, normalize=False)
    grid = (grid.cpu().float().numpy().transpose((1,2,0)) * 0.5 + 0.5) * 255.0
    return grid.astype(np.uint8)

# Li Jiaxin, define a function to convert log depth to uint8 img
# the log depth is single channel tensor ranges around [-10, 10]
def log_depth2im(image_tensor):
    image_numpy = image_tensor[0].cpu().float().numpy()
    minimum = np.amin(image_numpy)
    maximum = np.amax(image_numpy)
    image_numpy = (np.transpose(image_numpy, (1,2,0)) - minimum) / (maximum-minimum) * 255
    return image_numpy.astype(np.uint8).repeat(3,2)

def log_depth2grid_im(image_tensor):
    # the clamp is according to the data processing
    image_tensor = image_tensor.clamp(-0.84, 2.11)
    grid = torchvision.utils.make_grid(image_tensor, nrow=5, normalize=True)
    grid = grid.cpu().float().numpy().transpose((1, 2, 0)) * 255
    return grid.astype(np.uint8)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def stem_dir(name: str) -> str:
    """
    Extract base directory
    """
    return os.path.splitext(os.path.basename(name))[0]


# --- loading ckpt ---
def load_state_dict_strict_or_partial(module, state_dict, strict_first=True, tag=""):
    """
    Try strict load first (Case 1). If it fails, fallback to partial filtered load (Case 3).
    """
    if strict_first:
        try:
            module.load_state_dict(state_dict, strict=True)
            print(f"[Strict] Loaded {tag or module.__class__.__name__}")
            return True
        except RuntimeError as e:
            print(f"[Strict failed] {tag or module.__class__.__name__} -> fallback to partial")
            print("Reason:", e)

    model_dict = module.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape}
    print(f"[Partial] Loaded {len(filtered)}/{len(model_dict)} params for {tag or module.__class__.__name__}")
    model_dict.update(filtered)
    module.load_state_dict(model_dict, strict=True)
    return False


def model_state_dict_convert_auto(state_dict, gpu_ids=None):
    """
    让 ckpt 的 state_dict key 和当前运行方式匹配：
    - 多卡 DataParallel: key 需要 'module.' 前缀
    - 单卡/CPU: key 不需要 'module.' 前缀
    """
    if not state_dict:
        return state_dict

    # 默认：单卡（或 CPU）
    if gpu_ids is None:
        multi_gpu = False
    else:
        multi_gpu = len(gpu_ids) >= 2

    first_key = next(iter(state_dict))
    has_module = first_key.startswith("module.")

    # 已经匹配：不需要转换
    if has_module == multi_gpu:
        return state_dict

    # ckpt 是多卡、当前是单卡：去掉 module.
    if has_module and not multi_gpu:
        return {k[7:]: v for k, v in state_dict.items()}

    # ckpt 是单卡、当前是多卡：加上 module.
    return {"module." + k: v for k, v in state_dict.items()}


def load_detector(
    detector,
    ckpt_path: str,
    map_location: str = "cpu",
    strict_first: bool = True,
    tag: str = "detector",
):
    """
    Load detector weights from checkpoint.

    Supports ckpt formats:
      1) state_dict directly
      2) dict with key 'detector' -> state_dict
      3) other nested formats if your model_state_dict_convert_auto handles it
    """
    # ---- 1) load ckpt ----
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # ---- 2) pick detector state ----
    det_raw = ckpt["detector"] if isinstance(ckpt, dict) and "detector" in ckpt else ckpt

    # ---- 3) normalize keys (your existing helper) ----
    det_sd = model_state_dict_convert_auto(det_raw)

    # ---- 4) load (your existing helper) ----
    load_state_dict_strict_or_partial(
        detector.detector,       # NOTE: same as your original code
        det_sd,
        strict_first=strict_first,
        tag=tag,
    )

    return ckpt

# --- random seed ---

def setup_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for python, numpy, torch (cpu & cuda).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker_fn(base_seed: int):
    """
    Returns a worker_init_fn for DataLoader.
    """
    def _seed_worker(worker_id: int):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _seed_worker


def build_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def get_logger(
    level: str = "INFO",
    log_path=None,
    name=None,
    ) -> logging.Logger:

    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        # Initialize the logger
        level = logging.__dict__[level]  # TODO: fix this
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s|%(name)s|%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_path:
            # Add log file handler
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


import yaml



def load_experiment_config(name, cfg_path=None, default_name="best"):
    if cfg_path is None:
        cfg_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "experiments.yaml",
        )
        cfg_path = os.path.abspath(cfg_path)

    with open(cfg_path, "r") as f:
        cfgs = yaml.safe_load(f)

    # 1️⃣ exact match
    if name in cfgs:
        return cfgs[name]

    # 2️⃣ prefix before "_"
    if "_" in name:
        prefix = name.split("_")[0]
        if prefix in cfgs:
            print(f"[INFO] Experiment '{name}' matched config '{prefix}'.")
            return cfgs[prefix]

    # 3️⃣ fallback logic (keep your original)
    if default_name in cfgs:
        print(f"[WARN] Experiment '{name}' not found in {cfg_path}. "
              f"Falling back to '{default_name}'.")
        return cfgs[default_name]
    else:
        # 如果 best 也没写，就 fallback 到 D3
        if "D3" in cfgs:
            print(f"[WARN] Experiment '{name}' not found in {cfg_path}. "
                  f"Falling back to 'D3'.")
            return cfgs["D3"]

        raise ValueError(
            f"Experiment '{name}' not found in {cfg_path}, and no fallback "
            f"('{default_name}' or 'D3') exists. Available: {list(cfgs.keys())}"
        )

def override_opt_dict(opt, cfg: dict, verbose=True):
    for k, v in cfg.items():
        if hasattr(opt, k):
            old_v = getattr(opt, k)
            setattr(opt, k, v)
            if verbose and old_v != v:
                print(f"[Config] opt.{k}: {old_v} -> {v}")
        else:
            # 允许 opt 没有的字段（eval 阶段很有用）
            setattr(opt, k, v)
            if verbose:
                print(f"[Config] opt.{k}: (new) -> {v}")

import zlib
def make_pose_rng(base_seed: int, scene: str, qk: str):

    
    # scene + qk 稳定哈希，保证同 pose 同 scene 永远同采样
    h = zlib.adler32((scene + "|" + qk).encode("utf-8"))
    seed = (int(base_seed) + int(h)) % (2**32)

    return np.random.default_rng(seed)

