import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from util.gaussian_render import Camera, GaussianModel, Renderer


def build_cam_info(K, R_w2c, t_w2c, width=400, height=400) -> dict:
    K = K.detach().float().cpu()
    if t_w2c.ndim == 2:
        t_w2c = t_w2c.view(3)

    fx = float(K[0, 0].item())
    fy = float(K[1, 1].item())
    cx = float(K[0, 2].item())
    cy = float(K[1, 2].item())
    R_c2w = R_w2c.t()
    t_c2w = (-R_w2c.t() @ t_w2c)
    return {
        "width": width,
        "height": height,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "position": t_c2w.tolist(),
        "rotation": R_c2w.tolist(),
    }

def render_one(model: GaussianModel, K, R_w2c, t_w2c, width, height) -> np.ndarray:
    """
    Render RGB 
    """

    device = model.means3D.device
    camera = Camera()
    cam_info = build_cam_info(K, R_w2c, t_w2c, width, height)
    camera.load(cam_info)
    camera.bg = torch.ones(3, device=device)
    rgb,_,_, _ = Renderer(model, camera, logging=False).render()  # (3,H,W)
    # rgb: (3,H,W)
    rgb_np = rgb.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)   # (H,W,3)

    return rgb_np
        
def get_gs_model(gs_root: str, scene: str):

    class_id = scene.split("-")[0]   # e.g. 02691156
    ply_path = os.path.join(gs_root, class_id, scene + ".ply")

    gm = GaussianModel()
    gm.load(ply_path)  # 例如 gm.load_ply(...), gm.load_ckpt(...)

    return gm

def render_gt_est(opt,result,save_root, epoch):
    # general info
    name = result["name"]
    K = result["K"]
    width = result["width"]
    height = result["height"]

    # GT pose
    R_gt = result["R_gt"]
    t_gt = result["t_gt"]

    # 估计 pose
    R = result["R"]
    t = result["t"]
    device = result["K"].device
    R_est = torch.from_numpy(R).float().to(device)
    t_est = torch.from_numpy(t).float().to(device)
    
    # 误差
    R_err = result["R_err"]
    t_err = result["t_err"]

    # 匹配
    n_matches_gt = result["n_matches_gt"]
    n_matches_est = result["n_matches_est"]

    # PnP 内点
    n_inliers = result["n_inliers"]
    
    # load GS model
    scene = name.split("/")[0]
    gs_model = get_gs_model(opt.gs_root_path, scene)

    img_gt  = render_one(gs_model, K, R_gt, t_gt, width, height)
    img_est = render_one(gs_model, K, R_est, t_est, width, height)
    img_pair = np.concatenate([img_gt, img_est], axis=1)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_pair)
    plt.axis("off")

    # 图像下方文字（axes 坐标系）
    plt.text(
        0.25, -0.05, f"GT  ({n_matches_gt})",
        transform=plt.gca().transAxes,
        ha="center", va="top", fontsize=12
    )
    plt.text(
        0.75, -0.05, f"EST ({n_matches_est})",
        transform=plt.gca().transAxes,
        ha="center", va="top", fontsize=12
    )


    # plt.title(
    #     f"epoch={epoch+1}   "
    #     f"R_err={R_err:.2f}°   t_err={t_err:.2f}   "
    #     f"inliers={n_inliers}/{n_matches_est}   "
    # )

    # save_path = os.path.join(save_root, f"epoch_{epoch+1}.png")
    plt.title(
        f"name={epoch}   "
        f"R_err={R_err:.2f}°   t_err={t_err:.2f}   "
        f"inliers={n_inliers}/{n_matches_est}   "
    )

    # save_path = os.path.join(save_root, f"{epoch}.png")
    safe_name = name.replace("/", "_").split(".")[0]
    save_path = os.path.join(save_root, f"{safe_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    # plt.show()
