import sys
import argparse
import os
import glob
import torch
import random
import numpy as np
import yaml

from config.options import BaseOptions
from datasets.dataset_loader import init_data_loader
from models.gs_matcher import ModelMatcher
from evaluations.metrics import summarize_metrics
from util.util import get_logger, setup_seed, load_experiment_config
from config.path import dataset_dir, RUNS_DIR

logger = get_logger(level="INFO", name="gomatch_eval")


def eval_model(
    opt,
    data_loader,
    cache_dir,
    ckpt_path=None,
    oracle=False,
    debug=False,
    overwrite=False,
    prefix=None,
    split=None,
):
    if oracle:
        logger.info(f">>>>>Oracle Matching")
        save_path = os.path.join(cache_dir, f"OracleMatcher.npy")
    elif ckpt_path:
        logger.info(f">>>>>{ckpt_path}")
        save_path = os.path.join(
            cache_dir, f'{split}_{os.path.basename(ckpt_path)}'.replace("ckpt", "npy")
        )
    if debug:
        save_path = save_path.replace("npy", "debug.npy")
    if prefix:
        save_path = save_path.replace("npy", f"{prefix}.npy")

    # Initialize result dict
    if os.path.exists(save_path) and not overwrite:
        logger.info(f"Load cached results: {save_path}")
        metrics = np.load(save_path, allow_pickle=True).item()
        summarize_metrics(metrics, split=split)
        return metrics
    
    render_dir = os.path.join(cache_dir, 'render')
    os.makedirs(render_dir, exist_ok=True)

    # --- build/load model ---
    model = ModelMatcher(opt)  # 你现有构造函数是 ModelMatcher(opt)

    # load ckpt (if provided)
    if (not oracle) and (ckpt_path is not None):
        model.load_ckpt(ckpt_path, strict=True, load_optimizer=False)
    # --- run evaluation ---
    metrics = model.evaluate(data_loader, sc_thres=0.5, save_root=render_dir, debug=debug, oracle=oracle, render_dir=render_dir)
    


    # --- summarize + save ---
    np.save(save_path, metrics)
    summarize_metrics(metrics, split=split)
    logger.info(f"Save results to : {save_path}")


    return metrics

def run_benchmark(
    opt,
    odir,
    dataset_name,
    run_name,
    split="test",
    ckpt=None,
    oracle=False,
    debug=False,
    overwrite=False,
    prefix=None,
):
    assert ckpt or oracle

    # Load dataset
    if dataset_name == 'shapesplat':
        gs_root_path = str(dataset_dir("gs_root"))
        # opt.xfeat_root = os.path.join(gs_root_path, 'XFeat_kpt')
        # opt.match_root = os.path.join(gs_root_path, 'match_thresh0004')
        opt.xfeat_root = os.path.join(gs_root_path, 'XFeat_upgrade')
        opt.match_root = os.path.join(gs_root_path, 'match_new0004')
        opt.gs_root_path =gs_root_path
        print('evaluate on ShapeSplat dataset!')
    elif dataset_name == 'scared':
        print('evaluate on SCARED dataset!')
        root_path = str(dataset_dir("gs_root_scared"))
        opt.xfeat_root = os.path.join(root_path, 'scared_XFeat')
        opt.match_root = os.path.join(root_path, 'scared_match_thresh0004')       
        opt.gs_root_path =os.path.join(root_path, 'scared_splat')  
        
    data_loader = init_data_loader(opt, split=split)    

    # --- compute ransac_thres ONCE (K is constant in your dataset) ---
    e_px = 5.0  # fixed pixel threshold
    first_batch = next(iter(data_loader))
    if first_batch.get("name", None) is None:
        raise RuntimeError("First batch is empty (name is None). Cannot read K.")

    K0 = first_batch["K"][0]  # (3,3) torch
    fx = float(K0[0, 0].item())
    fy = float(K0[1, 1].item())
    f = 0.5 * (fx + fy)

    opt.ransac_thres = e_px / f




    # Cache dir
    tag =  f'{dataset_name}_{run_name}'
    cache_dir = os.path.join(odir, tag)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Eval
    eval_model(
        opt,
        data_loader,
        cache_dir,
        ckpt_path=ckpt,
        oracle=oracle,
        debug=debug,
        overwrite=overwrite,
        prefix=prefix,
        split=split,
    )
    
    
# def load_experiment_config(name, cfg_path=None):
#     if cfg_path is None:
#         cfg_path = os.path.join(
#             os.path.dirname(__file__),
#             "..",
#             "configs",
#             "experiments.yaml",
#         )
#         cfg_path = os.path.abspath(cfg_path)

#     with open(cfg_path, "r") as f:
#         cfgs = yaml.safe_load(f)

#     if name not in cfgs:
#         raise ValueError(
#             f"Experiment '{name}' not found in {cfg_path}. "
#             f"Available: {list(cfgs.keys())}"
#         )
#     return cfgs[name]

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,choices=["shapesplat","scared", ], default="shapesplat" )
    parser.add_argument("--splits", nargs="*", type=str, default=["test"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--name", type=str, default='A2')
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--odir", type=str, default="outputs/benchmark_cache_release")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    
    # 1) parse BaseOptions
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    opt = BaseOptions().parse()
    sys.argv = old_argv

    # 2) argparse 覆盖 BaseOptions
    override_opt_dict(opt, vars(args), verbose=True)
    
    # 3) YAML experiment 覆盖（最高优先级）
    if not args.oracle:
        exp_cfg = load_experiment_config(args.name, cfg_path='config/config_shapesplat.yaml')
        override_opt_dict(opt, exp_cfg)

    # ----- Random seed setup, so that align with 2D inference-----
    seed = opt.seed if opt.seed is not None else random.randint(0, 2**32 - 1)
    print(f"[INFO] Using random seed: {seed}")
    opt.seed = seed
    setup_seed(seed)
    
    # ---- Save ---
    args.odir = RUNS_DIR / 'eval'
    
    print(f"Splits : {args.splits}")
    for split in args.splits:
        print(f"\n\n Split = {split}")
        run_benchmark(
            opt,
            odir=args.odir,
            dataset_name=args.dataset,
            run_name=args.name,
            split=split,
            ckpt=args.ckpt,
            oracle=args.oracle,
            debug=args.debug,
            overwrite=args.overwrite,
            prefix=args.prefix,
        )
