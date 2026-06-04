from pathlib import Path
import yaml
from typing import Union

# 本文件位于 src/config/ 下，向上两级是仓库根
ROOT = Path(__file__).resolve().parents[2]

_DEFAULTS = {
    "split_dir": "cfg/split",
    "data_dir":  "data",
    "runs_dir":  "runs",
    "datasets":  {},
    "ckpt": {
        "external": "localdata/gs_mae_ckpts",
        "my": "localdata/my_ckpts",
        "pretrained_cfg": "ext/ShapeSplat-Gaussian_MAE/cfgs/finetune",
        "profile": None,
    },
}
PROFILES = {
    "modelnet10_1k": {
        "ckpt": "finetune_modelnet10_enc_full_group_xyz_1k_ckpt_best.pth",
        "cfg":  "finetune_modelnet10_enc_full_group_xyz_1k.yaml",
    },
    "modelnet10_4k": {
        "ckpt": "release_finetune_modelnet10_enc_full_4k_pretrain_1k_best_acc95.37.pth",
        "cfg":  "finetune_modelnet10_enc_full_group_xyz_4k.yaml",
    },
    "modelnet40_1k": {
        "ckpt": "finetune_modelnet40_enc_full_group_xyz_1k_ckpt_best.pth",
        "cfg":  "finetune_modelnet40_enc_full_group_xyz_1k.yaml",
    },
    "modelnet40_4k": {
        "ckpt": "release_finetune_modelnet40_enc_full_4k_pretrain_1k_best_acc93.43.pth",
        "cfg":  "finetune_modelnet40_enc_full_group_xyz_4k.yaml",
    },
}


def _load_cfg():
    cfg_file = ROOT / "cfg" / "paths.yaml"
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # 合并（ckpt 做细合并，避免整体被覆盖）
        merged = {**_DEFAULTS, **{k: v for k, v in user.items() if k != "ckpt"}}
        merged["ckpt"] = {**_DEFAULTS["ckpt"], **(user.get("ckpt") or {})}
        return merged
    return _DEFAULTS

_cfg = _load_cfg()

def _abs(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)

# 公开路径
SPLIT_DIR = _abs(_cfg.get("split_dir", "cfg/split"))
DATA_DIR  = _abs(_cfg.get("data_dir", "data"))
RUNS_DIR  = _abs(_cfg.get("runs_dir", "runs"))

_CKPT = _cfg["ckpt"]
CKPT_EXTERNAL_DIR  = _abs(_CKPT["external"])
CKPT_MY_DIR        = _abs(_CKPT["my"])
PRETRAINED_CFG_DIR = _abs(_CKPT["pretrained_cfg"]) 
CKPT_PROFILE_KEY   = _CKPT.get("profile")
if CKPT_PROFILE_KEY not in PROFILES:
    raise KeyError(f"Invalid ckpt.profile '{CKPT_PROFILE_KEY}'. "
                   f"Must be one of: {list(PROFILES)}")




# --------- 小工具函数 ---------

def external_ckpt_path() -> Path:
    """直接根据 profile 键解析 ckpt 路径"""
    fname = PROFILES[CKPT_PROFILE_KEY]["ckpt"]
    return (CKPT_EXTERNAL_DIR / fname).resolve()

def companion_yaml_path() -> Path:
    """直接根据 profile 键解析 cfg 路径"""
    fname = PROFILES[CKPT_PROFILE_KEY]["cfg"]
    return (PRETRAINED_CFG_DIR / fname).resolve()


def dataset_dir(name: str = "") -> Path:
    ds = _cfg.get("datasets") or {}
    if name and isinstance(ds, dict) and name in ds:
        return _abs(ds[name])
    # 没配 datasets 或没有该 name 时，回退到 DATA_DIR
    return DATA_DIR

def split_file(name: str) -> Path:
    p = (SPLIT_DIR / f"{name}.txt").resolve()
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}\nSPLIT_DIR={SPLIT_DIR}")
    return p

def my_ckpt(*parts) -> Path:
    """自己训练产出路径：my_ckpt('detector','epoch_10.pth')"""
    p = CKPT_MY_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def output_dir(*parts) -> Path:
    """返回并创建一个目录"""
    p = RUNS_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)   
    return p

