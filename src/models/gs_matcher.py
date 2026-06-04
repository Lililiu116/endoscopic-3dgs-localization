import os
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import time
from tqdm import tqdm


from models.gomatch.gomatch import OTMatcherDesc, OTMatcherDescCls
from models.gomatch.gomatch_pure import OTMatcher
from models.losses import compute_loss
from evaluations.metrics import compute_metrics_batch,compute_one_img
from evaluations.gs_renderer import render_gt_est

class ModelMatcher():
    """
    目标：把 GoMatch 的 Lightning 训练写法，改成你 detector 同款的“手写训练器”风格。
    你可以像用 ModelDetector 一样使用它：
      - set_input(data)
      - optimize() / test_model()
      - get_current_errors()
      - save_network()
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = torch.device(self.opt.device)

            
        # -------- build matcher --------
        if opt.matcher_type == 'Cls':
            self.matcher = OTMatcherDescCls(opt, d2=64, out_dim=128) 
            print('Using Cls Model!')
        elif opt.matcher_type == 'GoMatch':
            self.matcher = OTMatcher(
                p3d_type="bvs",
            )     
            print('Using Gomatch Model!')
        else:
            self.matcher = OTMatcherDesc(opt, d2=64, out_dim=128) 
        
        self.cls = isinstance(self.matcher, OTMatcherDescCls)

        # -------- move model to device --------
        self.matcher = self.matcher.to(self.device)

        # -------- optimizer / lr schedule --------
        self.optimizer = torch.optim.Adam(
            self.matcher.parameters(),
            lr=self.opt.learning_rate,
            betas=(0.9, 0.999),
        )
        # # shapesplat (huge)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode="min",
        #     factor=0.5,          # 可以降狠一点
        #     patience=3,          # 更快响应
        #     threshold=1e-4,
        #     threshold_mode="abs",
        #     cooldown=1,
        #     min_lr=1e-6,
        #     verbose=True,
        # )
        # scared (mideum)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=getattr(self.opt, "lr_factor", 0.8),
            patience=getattr(self.opt, "lr_patience", 8),
            threshold=getattr(self.opt, "lr_threshold", 1e-3),
            threshold_mode=getattr(self.opt, "lr_threshold_mode", "abs"),
            cooldown=getattr(self.opt, "lr_cooldown", 2),
            min_lr=getattr(self.opt, "min_lr", 1e-6),
            verbose=True,
        )
        # # small 
        # self.scheduler = None
            
        # --- state for logging ---
        self.data = None
        self.preds = None
        self.losses = {}
        self.metrics = {}
        self.iter = 0
        

        self.GPU_KEYS = {
            # matcher forward 必需
            "pts2dm", "idx2d", "desc2d", "pts3dm", "idx3d",
            # calculating loss
            "pts3d", "R","t", "pts2d"
        }
                    
    def to_cpu(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: self.to_cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self.to_cpu(i) for i in x]
        if isinstance(x, tuple):
            return tuple(self.to_cpu(i) for i in x)
        return x



    def set_input(self, data, epoch=None):
        self.data = {}
        for k, v in data.items():
            if torch.is_tensor(v) and (k in self.GPU_KEYS):
                self.data[k] = v.to(self.device, non_blocking=True)
            else:
                self.data[k] = v
        if self.opt.matcher_type == 'GoMatch':
            self.inputs = (
                self.data["pts2dm"],
                self.data["idx2d"],
                # self.data["desc2d"],
                # self.data["gs3d"],
                self.data["pts3dm"],
                self.data["idx3d"],
            )   
        else:         
            self.inputs = (
                self.data["pts2dm"],
                self.data["idx2d"],
                self.data["desc2d"],
                # self.data["gs3d"],
                self.data["pts3dm"],
                self.data["idx3d"],
            )
    def compute_loss_and_metrics(self):
        preds = self.matcher(*self.inputs)
        losses = compute_loss(
            self.data,
            preds,
            self.opt.inliers_only,
            cls=self.cls,
            rpthres=self.opt.rpthres,
        )

        # Compute metrics per step
        metrics = defaultdict(list)        
        with torch.no_grad():
            preds_cpu = self.to_cpu(preds)
            data_cpu = self.to_cpu(self.data)            
            
            raw_pose_errs = compute_metrics_batch(
                metrics, data_cpu, preds_cpu, cls=self.cls, sc_thres=0.5, ransac_thres=self.opt.ransac_thres
            )
        return preds, losses, metrics, raw_pose_errs

    
    @torch.no_grad()
    def forward_only(self):
        """只前向（验证/可视化用）"""
        self.matcher.eval()
        preds = self.matcher(*self.inputs)
        return preds

    def optimize(self, epoch=None):
        """训练一步：forward -> loss -> backward -> step"""
        self.matcher.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Compute loss and metrics
        preds, losses, metrics, _ = self.compute_loss_and_metrics()
        self.preds = preds
        loss = losses["loss"]

        loss.backward()
        self.optimizer.step()


        # 转成纯 float，便于 log/平均
        losses_f = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
        metrics_f = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}

        # Lightning 的 training_step return loss（你可以也返回）
        return losses_f, metrics_f

    

    def get_current_errors(self):
        """给你 main 循环里 tepoch.set_postfix 用"""
        out = {}
        out.update(self.losses)
        # metrics 你想显示哪些就加哪些
        for k in ["R_err", "t_err", "inliers", "recall", "precision"]:
            if k in self.metrics:
                out[k] = self.metrics[k]
        return out

    @torch.no_grad()
    def validate_one_epoch(self, val_loader, epoch:int=0, return_render_sample:bool=True):
        self.matcher.eval()

        sum_losses = defaultdict(float)
        cnt_losses = defaultdict(int)

        # 用来做 quantile
        all_pose_errs = []  # list of dicts: {"R_err": (B,), "t_err": (B,)}

        render_sample = None  
        
        for data in val_loader:
            if data.get("name", None) is None:
                continue
            
            self.set_input(data)

            
            # 抓一个样本用于后续 render（只抓第一个有效 batch）
            if return_render_sample and render_sample is None:
                render_sample = data

            # forward

            preds = self.matcher(*self.inputs)
                        

            # loss
            losses = compute_loss(
                self.data,
                preds,
                self.opt.inliers_only,   # <- 跟 train 对齐
                cls=self.cls,
                rpthres=self.opt.rpthres,
            )
            for k, v in losses.items():
                vv = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                sum_losses[k] += vv
                cnt_losses[k] += 1

            # metrics + pose errs
            metrics = defaultdict(list)
            preds_cpu = self.to_cpu(preds)
            data_cpu = self.to_cpu(self.data)       
            raw_pose_errs = compute_metrics_batch(
                metrics, data_cpu, preds_cpu, cls=self.cls, sc_thres=0.5, ransac_thres=self.opt.ransac_thres
            )
            if raw_pose_errs is not None:
                # raw_pose_errs 里一般是 tensor(B,)
                all_pose_errs.append({
                    "R_err": raw_pose_errs["R_err"].detach().cpu(),
                    "t_err": raw_pose_errs["t_err"].detach().cpu(),
                })
                
                


        # mean losses
        mean_losses = {k: (sum_losses[k] / max(1, cnt_losses[k])) for k in sum_losses}

        # quantiles
        quant = {}
        if len(all_pose_errs) > 0:
            errors_R = torch.cat([x["R_err"] for x in all_pose_errs], dim=0)
            errors_t = torch.cat([x["t_err"] for x in all_pose_errs], dim=0)

            for name, errors in [("R_err", errors_R), ("t_err", errors_t)]:
                valid = errors[errors > -1]
                if valid.numel() > 0:
                    q = torch.quantile(valid, torch.tensor([0.25, 0.5, 0.75]))
                    quant[f"{name}_q1"] = float(q[0].item())
                    quant[f"{name}_q2"] = float(q[1].item())
                    quant[f"{name}_q3"] = float(q[2].item())
                else:
                    quant[f"{name}_q1"] = -1.0
                    quant[f"{name}_q2"] = -1.0
                    quant[f"{name}_q3"] = -1.0

            quant["failed"] = int((errors_R == -1).sum().item())
        else:
            # 一个都没有就给默认
            quant = dict(R_err_q1=-1.0, R_err_q2=-1.0, R_err_q3=-1.0,
                        t_err_q1=-1.0, t_err_q2=-1.0, t_err_q3=-1.0,
                        failed=-1)
        if return_render_sample:
            return mean_losses, quant, render_sample
        return mean_losses, quant
    
    
    
    @torch.no_grad()
    def evaluate(self, loader, save_root, sc_thres=0.5, debug=False, oracle=False, render_dir =None):
        self.matcher.eval()
        metrics = defaultdict(list)
        metrics["n_queries"] = len(loader.dataset)
        rendered_count = 0
        max_render = 100

        t_start = time.time()
        for it, data in enumerate(tqdm(loader, desc=f"Eval oracle={oracle}", unit="batch")):
            if data.get("name", None) is None:
                continue

            self.set_input(data)   
            if not oracle:
                # Matching
                t0 = time.time()
                preds = self.matcher(*self.inputs)
                metrics["match_time"].append(time.time() - t0)
            else:
                preds = None
            preds_cpu = self.to_cpu(preds)
            data_cpu = self.to_cpu(self.data) 


            compute_metrics_batch(
                metrics, data_cpu, preds_cpu, 
                cls=self.cls, sc_thres=sc_thres, 
                ransac_thres=self.opt.ransac_thres,
                is_test=True
            )
            
            render_infos =  compute_one_img(
                self.data, preds, cls=self.cls, sc_thres=0.5, ransac_thres=self.opt.ransac_thres
            )
            
            for i, render_info in enumerate(render_infos):

                name = render_info.get("name", f"img_{i}")

                if render_info.get("failed", True):
                    print(f"[RENDER-SKIP] name={name} | PnP failed")
                    continue

                render_gt_est(
                    self.opt,
                    render_info,
                    save_root,
                    epoch=name   # ⭐ 用 name
                )

        if (it + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"[EVAL] it={it+1}/{len(loader)} elapsed={elapsed/60:.1f} min")

        return metrics

    
    @torch.no_grad()
    def render_one_example(self, data, save_root, epoch: int = 0):
        """独立渲染函数：给一个 batch（或样本）就输出可视化"""
        if data is None or data.get("name", None) is None:
            return
        self.set_input(data)
        self.matcher.eval()
        preds = self.matcher(*self.inputs)

        # render for visualization
        render_info =  compute_one_img(
                self.data, preds, cls=self.cls, sc_thres=0.5, ransac_thres=self.opt.ransac_thres, render_one = True
            )
        if render_info.get("failed", True):
            print(f"[RENDER-SKIP] epoch={epoch} name={render_info.get('name')} | PnP failed")
            return
        render_gt_est(self.opt, render_info,save_root,  epoch=epoch)
            
    # ---------------- checkpoint ----------------
    def save_ckpt(self, path, epoch, global_step, metrics=None, best=None, monitor=None):
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        ckpt = {
            "state_dict": self.matcher.state_dict(),
            "optimizer_states": [self.optimizer.state_dict()],
            "epoch": epoch,
            "global_step": global_step,
            "hparams": vars(self.opt),
            "metrics": metrics or {},
            "best": best or {},                 # <--- add
            "monitor": monitor or {},           # <--- add (optional, helpful)
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(ckpt, str(path))



    def load_ckpt(self, path, strict=True, load_optimizer=True):
        ckpt = torch.load(str(path), map_location="cpu")
        self.matcher.load_state_dict(ckpt["state_dict"], strict=strict)

        if load_optimizer and "optimizer_states" in ckpt and len(ckpt["optimizer_states"]) > 0:
            self.optimizer.load_state_dict(ckpt["optimizer_states"][0])

        if load_optimizer and self.scheduler is not None and ckpt.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)
        best = ckpt.get("best", {})
        return start_epoch, global_step, ckpt, best
