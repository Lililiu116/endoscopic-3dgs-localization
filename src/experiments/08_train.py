import random
import math
from tqdm import tqdm
import time
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import shutil

from config.path import dataset_dir, RUNS_DIR
from datasets.dataset_loader import init_data_loader
from config.matcher_options import MatcherOptions
from models.gs_matcher import ModelMatcher
from util.util import setup_seed, load_experiment_config, override_opt_dict

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def main(opt):

    # ----- Random seed setup -----
    seed = opt.seed if opt.seed is not None else random.randint(0, 2**32 - 1)
    print(f"[INFO] Using random seed: {seed}")
    opt.seed = seed
    setup_seed(seed)
    
    # ----- paths -----
    if opt.dataset == 'shapesplat':
        print('train on ShapeSplat dataset!')
        gs_root_path = str(dataset_dir("gs_root"))
        # xfeat_root = os.path.join(gs_root_path, 'XFeat_kpt')
        # match_root = os.path.join(gs_root_path, 'match_thresh0004')
        # query based
        xfeat_root = os.path.join(gs_root_path, 'XFeat_upgrade')
        match_root = os.path.join(gs_root_path, 'match_new0004')
        opt.gs_root_path =gs_root_path
        opt.xfeat_root =xfeat_root
        opt.match_root =match_root
    elif opt.dataset =='scared':
        print('train on SCARED dataset!')
        root_path = str(dataset_dir("gs_root_scared"))
        opt.xfeat_root = os.path.join(root_path, 'scared_XFeat')
        opt.match_root = os.path.join(root_path, 'scared_match_thresh0004')       
        opt.gs_root_path =os.path.join(root_path, 'scared_splat')          

    # ---- save ----
    resume_dir = getattr(opt, "resume_dir", None)
    if resume_dir is not None:
        resume_root = Path(resume_dir)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_root = RUNS_DIR / opt.exp / f"{opt.name}_{ts}"
        shutil.copytree(resume_root, exp_root, dirs_exist_ok=False)
        print(f"[RESUME-COPY] Copied {resume_root} -> {exp_root}")
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{opt.name}_{ts}"
        exp_root = RUNS_DIR / opt.exp / exp_name
        exp_root.mkdir(parents=True, exist_ok=True)
    log_dir = exp_root
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TB] Logging to: {log_dir}")
    img_dir = exp_root / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_root/ 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.ckpt"
    best_ckpt = ckpt_dir / "best.ckpt"
    

    # ---- Matcher model ----
    model = ModelMatcher(opt) 

    # --- resume ---
    
    best_Rq2 = float("inf")
    best_val = float("inf")  
    start_epoch = 0
    global_step = 0
    max_epoch  = getattr(opt, "max_epoch", 50)
    save_every = getattr(opt, "save_every", 10)

    finetune_ckpt = getattr(opt, "finetune_ckpt", None)
    finetune_strict = getattr(opt, "finetune_strict", False)
    # new_lr = 3e-5
    if resume_dir is not None:
        start_epoch, global_step, ckpt_last, best_dict = model.load_ckpt(
            last_ckpt, strict=True, load_optimizer=True
        )

        # 从 last.ckpt 里恢复 best（如果存在）
        best_Rq2 = best_dict.get("best_Rq2", best_Rq2)

    elif finetune_ckpt is not None:
        print(f"[FINETUNE] Init model from: {finetune_ckpt}")

        # 只加载模型，不加载 optimizer/scheduler
        model.load_ckpt(finetune_ckpt, strict=finetune_strict, load_optimizer=False)

        # ⭐关键：重置 best
        start_epoch = 0
        global_step = 0
        best_Rq2 = float("inf")

        # ⭐关键：重新设置 scheduler 状态（防止继承旧的 plateau 内部计数）
        if model.scheduler is not None:
            model.scheduler._reset()

    else:
        print("# Training from scratch.")
        start_epoch = 0
        global_step = 0
        best_Rq2 = float("inf")


    print(f"[RESUME] start_epoch={start_epoch}, global_step={global_step}, best_Rq2={best_Rq2}")

    # 1) dataset
    if opt.dataset == 'shapesplat':
        trainDataLoader = init_data_loader(opt, split='train')
        valDataLoader = init_data_loader(opt, split='val')    
    elif opt.dataset == 'scared':
        trainDataLoader = init_data_loader(opt, split='scared_train')
        valDataLoader = init_data_loader(opt, split='scared_val')        
    
    # --- compute ransac_thres ONCE (K is constant in your dataset) ---
    e_px = 5.0  # fixed pixel threshold
    first_batch = next(iter(trainDataLoader))
    if first_batch.get("name", None) is None:
        raise RuntimeError("First batch is empty (name is None). Cannot read K.")

    K0 = first_batch["K"][0]  # (3,3) torch
    fx = float(K0[0, 0].item())
    fy = float(K0[1, 1].item())
    f = 0.5 * (fx + fy)

    opt.ransac_thres = e_px / f

    
    for epoch in range(start_epoch, max_epoch):  
        # ---- accumulate for on_epoch=True ----
        sum_losses = defaultdict(float)
        cnt_losses = defaultdict(int)
        sum_metrics = defaultdict(float)
        cnt_metrics = defaultdict(int)
        
        with tqdm(trainDataLoader,desc=f"Epoch {epoch+1}/{max_epoch}", unit="batch",leave=True,) as tepoch:

            for i, data in enumerate(tepoch):
           
                if data.get("name", None) is None:
                    continue
                model.set_input(data, epoch=epoch) 
                losses_f, metrics_f = model.optimize() 
                # ---- TensorBoard: per-step ----
                # losses
                for k, v in losses_f.items():
                    sum_losses[k] += v
                    cnt_losses[k] += 1

                # metrics
                for k, v in metrics_f.items():
                    if v > -1:
                        sum_metrics[k] += v
                        cnt_metrics[k] += 1

                global_step += 1
                # tqdm 上显示当前 step 的 loss（只显示，不写 TB）
                tepoch.set_postfix(loss=losses_f.get("loss", float("nan")))

        # ---- epoch end: write TB once (on_epoch=True) ----
        for k in sum_losses:
            writer.add_scalar(f"train/{k}", sum_losses[k] / cnt_losses[k], epoch)

        for k in sum_metrics:
            v = (sum_metrics[k] / cnt_metrics[k]) if cnt_metrics[k] > 0 else -1.0
            writer.add_scalar(f"train/{k}", v, epoch)

        writer.add_scalar("train/epoch", epoch, epoch)   

        # ============== VALIDATION ==============
        val_losses, val_quant, render_sample = model.validate_one_epoch(valDataLoader, epoch=epoch, return_render_sample=True)

        # write val losses
        for k, v in val_losses.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        # write val quantiles
        for k, v in val_quant.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        # =========== modify learning rate ==============
        # monitor_val = val_losses.get("loss", None)
        monitor_val = val_quant.get("R_err_q2", None)
        if model.scheduler is not None and monitor_val is not None:
            model.scheduler.step(float(monitor_val))

        lr_now = model.optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", lr_now, epoch)


        # ---------------- CHECKPOINT (Lightning style) ----------------
        # 1) always save last.ckpt every epoch (overwrite)
        model.save_ckpt(
            last_ckpt,
            epoch=epoch,
            global_step=global_step,
            metrics={"val_losses": val_losses, "val_quant": val_quant},
            best={"best_Rq2": best_Rq2},
        )
        # 3) save periodic checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            periodic_ckpt = ckpt_dir / f"epoch_{epoch+1}.ckpt"
            model.save_ckpt(
                periodic_ckpt,
                epoch=epoch,
                global_step=global_step,
                metrics={"val_losses": val_losses, "val_quant": val_quant},
                best={"best_Rq2": best_Rq2},
            )
            print(f"[CKPT] Saved periodic checkpoint: {periodic_ckpt}")

        # 2) save best.ckpt only if val/loss improves
        cur = val_quant.get("R_err_q2", None)
        if cur is not None and cur > 0 and cur < best_Rq2:
            best_Rq2 = float(cur)

            model.save_ckpt(
                best_ckpt,
                epoch=epoch,
                global_step=global_step,
                metrics={"val_losses": val_losses, "val_quant": val_quant},
                best={"best_Rq2": best_Rq2},  # ⭐加上
            )

            model.render_one_example(render_sample, save_root=img_dir, epoch=epoch)
            print(f"[CKPT] New best (monitor=val/R_err_q2): {best_Rq2:.4f} -> {best_ckpt}")


        # print
        print(f"[VAL] epoch={epoch} val_loss={val_losses.get('loss', None)} lr={lr_now} "
              f"R_q2={val_quant.get('R_err_q2', None)} t_q2={val_quant.get('t_err_q2', None)} "
              f"failed={val_quant.get('failed', None)}")
        


    writer.close()


    
if __name__ == '__main__':
    # sh > yaml > option
    opt = MatcherOptions().parse()
    exp_cfg = load_experiment_config(opt.name, cfg_path='config/config_shapesplat.yaml')
    override_opt_dict(opt, exp_cfg)


    
    main(opt)