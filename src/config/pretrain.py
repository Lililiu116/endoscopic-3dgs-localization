# src/config/pretrain.py
from pathlib import Path
import yaml
from typing import Dict, Any
from . import path as P

def _read_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def companion_yaml_path() -> Path:
    return P.companion_yaml_path()

def read_model_params() -> Dict[str, Any]:
    """
    直接读取预训练 yaml 的模型参数：
    - 优先返回 data['model']；
    - 若顶层也有同名键，则以 model 块为准进行合并。
    """
    data = _read_yaml(companion_yaml_path())
    top  = data if isinstance(data, dict) else {}
    model = top.get("model", {}) if isinstance(top.get("model", {}), dict) else {}

    # 合并：顶层 -> model（model 覆盖顶层）
    merged = dict(top)
    merged.update(model)
    # 去除非模型字段的明显干扰（可选）
    for k in ("model", "optimizer", "scheduler", "dataset", "ckpt", "runs_dir", "split_dir", "data_dir", "datasets"):
        merged.pop(k, None)
    return merged
