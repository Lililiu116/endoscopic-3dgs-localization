import numpy as np
import os
import ntpath
import time
from . import util
import visdom
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d
from util.vis_tools import plot_cage_visdom, plot_cage
import math
class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = 1
                
        self.env = f"{opt.name}_{int(time.time())}"

        self.win_size = 256
        self.name = opt.name
        server = f'http://G27ws{opt.server:04d}-Linux'

        # if self.display_id > 0:
            
        #     self.vis = visdom.Visdom(env='%d' % self.display_id)
        #     # self.vis = visdom.Visdom(server='http://127.0.0.1', port=8097, env='%d' % self.display_id)

        # 关闭代理环境变量，避免被代理拦截
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''

        # 创建 visdom 实例，跳过 socket 连接（防止被代理阻挡）
        if opt.save:
            self.vis = visdom.Visdom(server=server, port=8097, use_incoming_socket=False,env=self.env)
        else:
            self.vis = visdom.Visdom(server=server, port=8097, use_incoming_socket=False)

        if self.vis.check_connection():
            print("Visdom 连接成功")
        else:
            print("Visdom 连接失败")

        print('Visdom connected:', self.vis.check_connection())
        


    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, iter=0):
        if self.display_id > 0: # show images in the browser
            idx = 1
            for label, item in visuals.items():
                if 'data_vis' in label:
                    data_vis_np, data_vis_color_np = item
                    self.vis.scatter(data_vis_np,
                                     Y=None,
                                     opts=dict(title=label,
                                               markersize=2,
                                               markercolor=data_vis_color_np,
                                               markersymbol='circle'),
                                     win=self.display_id + idx,
                                     name='data_vis')

                elif 'img' in label:
                    # the transpose: HxWxC -> CxHxW
                    self.vis.image(np.transpose(item, (2,0,1)), opts=dict(title=label),
                                   win=self.display_id + idx)
                elif "recon" in label:
                    pts_np, color_np = item        # pts_np: (N,3), color_np: (N,3)       
                    
                    # ---- 1. 规范颜色到 [0, 255]，并保证是 numpy ndarray ----
                    color_np = np.asarray(color_np)

                    cmin, cmax = color_np.min(), color_np.max()

                    if cmin >= -1.0 and cmax <= 1.0:
                        # 例如 SH 处理后的颜色在 [-1,1]
                        color_01 = (color_np + 1.0) / 2.0
                        color_01 = np.clip(color_01, 0.0, 1.0)
                        color_vis = (color_01 * 255.0).astype(np.int32)   # N x 3
                    elif cmin >= 0.0 and cmax <= 1.0:
                        # 已经在 [0,1]
                        color_vis = (np.clip(color_np, 0.0, 1.0) * 255.0).astype(np.int32)
                    else:
                        # 当成已经是 [0,255]
                        color_vis = np.clip(color_np, 0, 255).astype(np.int32)

                    # ---- 2. 传给 visdom ----
                    self.vis.scatter(
                        pts_np,                      # (N,3) numpy
                        Y=None,
                        opts=dict(
                            title=label,
                            markersize=2,
                            markercolor=color_vis,   # (N,3) numpy → 每点 RGB 颜色
                            markersymbol='circle',
                        ),
                        win=self.display_id + idx,
                        name='recon',
                    )



                elif 'cage_vis' in label:
                    cage   = item['cage']
                    points = item['points']
                    faces  = item['faces']
                    title  = item.get('title', label)
                    
                    try:
                        win_id = plot_cage_visdom(
                            vis=self.vis,
                            cage=cage,
                            points=points,
                            faces=faces,
                            title=title,
                            env=getattr(self.vis, 'env', 'main'),
                            win=f'{self.display_id}_{label}',
                        )

                    except Exception as e:
                        print('[VIS][ERROR] cage_vis failed:', e)


                idx += 1

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, errors):
        # clamp the errors at plot, to increase resolution
        # for key, value in errors.items():
        #     if value > 1:
        #         errors[key] = 1

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
        
        

    # def save_current_results(self, visuals, epoch, iter=0, save_dir: Path = None):
    #     save_dir.mkdir(parents=True, exist_ok=True)   # Path.mkdir，确保目录存在
    #     for label, item in visuals.items():
    #         # if 'data_vis' in label or 'pc' in label:
    #         if 'dst_data_vis' or 'recon'in label:
    #             pts, cols = item
    #             cols = cols.astype(np.float32)
    #             if cols.max() > 1.0:
    #                 cols /= 255.0

    #             fig = plt.figure(figsize=(5, 5))
    #             ax = fig.add_subplot(111, projection="3d")
    #             ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=2)
    #             ax.axis("off")

    #             out_path = save_dir / f"ep{epoch:03d}_it{iter:06d}_{label}.png"  # 用 /
    #             plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
    #             plt.close(fig)
    

    def save_current_results(self, visuals, epoch, iter=0, save_dir: Path = None):
        save_dir.mkdir(parents=True, exist_ok=True)

        # -------- fixed layout --------
        ordered_labels = [
            "src_data_vis",        "pcn_recon_input",     "pcn_recon_complete",
            "dst_data_vis",        "pcn_recon_coarse",    "pcn_recon_fine",
        ]

        # -------- create 2×3 panel --------
        fig, axes = plt.subplots(
            2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"}
        )
        axes = axes.flatten()

        for ax, label in zip(axes, ordered_labels):

            if label not in visuals:
                ax.set_title(label + " (missing)")
                ax.axis("off")
                continue

            pts, cols = visuals[label]

            cols = np.asarray(cols, dtype=np.float32)
            if cols.max() > 1.0:
                cols = cols / 255.0

            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=3)
            ax.set_title(label, fontsize=11)
            ax.axis("off")

        # save final panel
        out_path = save_dir / f"ep{epoch:03d}_it{iter:06d}_panel.png"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)



    def save_current_results_ply(self, visuals, epoch, iter=0, save_dir: Path = None):
        save_dir.mkdir(parents=True, exist_ok=True)
        for label, item in visuals.items():
            if 'data_vis' in label or 'pc' in label:
                pts, cols = item
                pts = np.asarray(pts, dtype=np.float32)
                cols = np.asarray(cols, dtype=np.float32)
                if cols.max() > 1.0:  # 如果是 0–255，归一化到 0–1
                    cols /= 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)

                out_path = save_dir / f"ep{epoch:03d}_it{iter:06d}_{label}.ply"
                o3d.io.write_point_cloud(str(out_path), pcd)
                
                

    def save_current_errors(self, epoch, counter_ratio, errors, save_dir: Path = None, fname="loss.png"):
        """
        离线版本：用 matplotlib 把 loss 曲线保存到 PNG。
        每次调用都会覆盖同名文件，达到“持续更新”的效果。
        """
        if save_dir is None:
            save_dir = Path("plots")
        save_dir.mkdir(parents=True, exist_ok=True)

        x = float(epoch) + float(counter_ratio)

        # 初始化缓存
        if not hasattr(self, "_plt_cache"):
            self._plt_cache = {
                "legend": list(errors.keys()),
                "X": [],
                "Y": {k: [] for k in errors.keys()},
            }

        # 新增 step
        self._plt_cache["X"].append(x)
        for k, v in errors.items():
            if k not in self._plt_cache["Y"]:
                self._plt_cache["legend"].append(k)
                self._plt_cache["Y"][k] = []
            self._plt_cache["Y"][k].append(float(v))

        # 绘制
        plt.figure(figsize=(8, 5), dpi=140)
        for k in self._plt_cache["legend"]:
            plt.plot(self._plt_cache["X"], self._plt_cache["Y"][k], label=k)
        plt.title(f"{getattr(self, 'name', 'training')} loss over time")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存
        png_path = save_dir / fname
        plt.savefig(png_path)
        plt.close()
