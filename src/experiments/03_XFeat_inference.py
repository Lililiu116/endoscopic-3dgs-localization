import os
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
import torch
import h5py
import hashlib
import torch.nn.functional as F
import glob
import re
from scipy.spatial import cKDTree
import cv2

from util.gaussian_render import Camera, GaussianModel, Renderer
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """
    Suppress BOTH Python prints and C/C++ prints to stdout/stderr inside the context.
    Works by redirecting file descriptors 1 and 2.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)
# ----------------------------
# Visualization
# ----------------------------
def show_render_with_kpts(imgs, kpts_list, view_indices, point_size=8):
    """Blocking visualization: close window to continue."""
    for img, kpts, vid in zip(imgs, kpts_list, view_indices):
        title = f"view_{vid:03d}"

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(img)
        axes[0].set_title(f"{title} | Render")
        axes[0].axis("off")

        axes[1].imshow(img)
        axes[1].scatter(kpts[:, 0], kpts[:, 1], s=point_size)
        axes[1].set_title(f"{title} | XFeat kpts ({len(kpts)})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()


# ----------------------------
# XFeat
# ----------------------------
def load_xfeat(device="cuda"):
    try:
        import sys
        sys.path.append('/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/ext/XFeat')
        from modules.xfeat import XFeat
        xfeat = XFeat()
    except Exception:
        xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True)
        print("[INFO] Loaded XFeat from torch.hub")

    return xfeat.to(device).eval()


def _to_tensor_img(img: np.ndarray, device="cuda") -> torch.Tensor:
    """img: HxWx3 float(0..1) or uint8(0..255) -> 1x3xHxW float32"""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    if t.max() > 1.5:
        t = t / 255.0
    return t.to(device)


def run_xfeat(xfeat, imgs, top_k=1024, device="cuda"):
    """
    Returns list of dicts:
      - kpt:  (N,2) float32
      - feat: (N,D) float32
      - conf: (N,) float32
    """
    results = []
    for img in imgs:
        t = _to_tensor_img(img, device=device)
        with torch.no_grad():
            out = xfeat.detectAndCompute(t, top_k=top_k)

        d = out[0]
        kpts = d["keypoints"]
        desc = d["descriptors"]
        conf = d["scores"]

        # strip batch dim if exists
        if kpts.ndim == 3:
            kpts = kpts[0]
        if desc.ndim == 3:
            desc = desc[0]
        if conf.ndim == 2:
            conf = conf[0]

        results.append({
            "kpt":  kpts.detach().cpu().numpy().astype(np.float32).reshape(-1, 2),
            "feat": desc.detach().cpu().numpy().astype(np.float32),
            "conf": conf.detach().cpu().numpy().astype(np.float32).reshape(-1),
        })
    return results


# ----------------------------
# Camera / Rendering
# ----------------------------
def convert_cam_coords(c2w: np.ndarray) -> np.ndarray:
    """
    Apply your coordinate conversion once here.
    """
    C = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)
    return c2w @ C


def camera_info_from_frame(frame, camera_angle_x, width=400, height=400):
    c2w_raw = np.array(frame["transform_matrix"], dtype=np.float32)
    c2w = convert_cam_coords(c2w_raw)

    R_c2w = c2w[:3, :3]
    Cpos = c2w[:3, 3]

    fx = width / (2.0 * np.tan(camera_angle_x / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return {
        "width": width,
        "height": height,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "position": Cpos.tolist(),
        "rotation": R_c2w.tolist(),
    }




def render_one_rgbdw(model: GaussianModel, cam_info: dict, weight_eps: float = 1e-6):
    """
    Render RGB + expected depth (Dep/Wei) + weight (Wei).
    weight is accumulated alpha*T (≈ 1 - final_T), NOT per-Gaussian alpha.
    """

    device = model.means3D.device
    camera = Camera()
    camera.load(cam_info)
    camera.bg = torch.ones(3, device=device)
    rgb,_,depth_hw, weight_hw = Renderer(model, camera, logging=False).render()  # (3,H,W)
    # rgb: (3,H,W), depth_hw: (H,W), alpha_hw: (H,W)
    
    # depth might be undefined where weight is tiny
    depth_hw = torch.where(weight_hw > weight_eps, depth_hw, torch.zeros_like(depth_hw))

    rgb_np = rgb.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)   # (H,W,3)
    depth_np = depth_hw.detach().cpu().numpy().astype(np.float32)               # (H,W)
    weight_np = weight_hw.detach().cpu().numpy().astype(np.float32)               # (H,W)
    return rgb_np, depth_np, weight_np

# ----------------------------
# List + path resolving
# ----------------------------
def read_list(txt_path: str):
    items = []
    prefixes = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def category_from_filename(ply_name: str):
    # expected: 04379243-xxxx.ply  -> category = 04379243
    base = os.path.basename(ply_name)
    cat = base.split("-", 1)[0]
    return cat if cat.isdigit() else None


def resolve_model_path(shape_splat_root: str, ply_name: str):
    """
    Most cases: ShapeSplat/<category>/<ply_name>
    Fallback: recursive search (slower).
    """
    ply_name = os.path.basename(ply_name)
    cat = category_from_filename(ply_name)

    if cat is not None:
        cand = os.path.join(shape_splat_root, cat, ply_name)
        if os.path.exists(cand):
            return cand, cat

    # fallback (if list contains weird names or category folder differs)
    for root, _, files in os.walk(shape_splat_root):
        if ply_name in files:
            found = os.path.join(root, ply_name)
            # try infer category from parent
            cat2 = os.path.basename(os.path.dirname(found))
            return found, cat2

    return None, cat


def stable_scene_seed(base_seed: int, scene_id: str) -> int:
    h = hashlib.sha1(f"{base_seed}_{scene_id}".encode("utf-8")).hexdigest()[:8]
    return (base_seed + int(h, 16)) % (2**31 - 1)


def sample_view_indices(num_views: int, K: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_views, size=min(K, num_views), replace=False)
    return np.sort(idx)

# ----------------------------
# Saving
# ----------------------------
def save_scene_h5(out_root, category, scene_id, model_path,
                  view_indices, cam_infos, results, split_name="train"):
    out_dir = os.path.join(out_root, category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene_id}.h5")

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["model_path"] = model_path
        # meta.attrs["split"] = split_name
        meta.create_dataset("view_indices", data=np.array(view_indices, dtype=np.int32))

        qroot = f.create_group("queries")
        for j, (vid, cam, res) in enumerate(zip(view_indices, cam_infos, results)):
            g = qroot.create_group(f"q_{j:04d}")

            # --- raw 2D (full) ---
            kptuv = np.asarray(res["kptuv"], dtype=np.float32)     # (N,2)
            feat  = np.asarray(res["feat"], dtype=np.float32)      # (N,D)
            conf  = np.asarray(res["conf"], dtype=np.float32)      # (N,)

            # --- reliable 3D subset ---
            kptxyz  = np.asarray(res["kptxyz"], dtype=np.float32)  # (M,3)
            xyz_idx = np.asarray(res["xyz_idx"], dtype=np.int64)   # (M,)

            g.create_dataset("kptuv", data=kptuv, compression="gzip", compression_opts=4)
            g.create_dataset("feat",  data=feat,  compression="gzip", compression_opts=4)
            g.create_dataset("conf",  data=conf,  compression="gzip", compression_opts=4)

            g.create_dataset("kptxyz",  data=kptxyz,  compression="gzip", compression_opts=4)
            g.create_dataset("xyz_idx", data=xyz_idx, compression="gzip", compression_opts=4)

            # attrs (both counts are useful)
            g.attrs["n_kptuv"]  = int(kptuv.shape[0])
            g.attrs["n_kptxyz"] = int(kptxyz.shape[0])
            g.attrs["view_index"] = int(vid)

            g.attrs["width"] = int(cam["width"])
            g.attrs["height"] = int(cam["height"])

            # extrinsics + intrinsics
            g.create_dataset("position", data=np.array(cam["position"], dtype=np.float32))
            g.create_dataset("rotation", data=np.array(cam["rotation"], dtype=np.float32))

            intr = np.array([[cam["fx"], 0, cam["cx"]],
                             [0, cam["fy"], cam["cy"]],
                             [0, 0, 1]], dtype=np.float32)
            g.create_dataset("intrinsics", data=intr)

    return out_path

def bilinear_sample_hw(img_hw: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    img_hw: (H,W)
    u,v: (N,) float pixel coords
    return: (N,) bilinear sampled
    """
    img = np.asarray(img_hw)
    H, W = img.shape

    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    wa = (x1.astype(np.float32) - u) * (y1.astype(np.float32) - v)
    wb = (u - x0.astype(np.float32)) * (y1.astype(np.float32) - v)
    wc = (x1.astype(np.float32) - u) * (v - y0.astype(np.float32))
    wd = (u - x0.astype(np.float32)) * (v - y0.astype(np.float32))

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def backproject_kpts_to_world(
    kpts_uv,
    depth_hw,
    weight_hw,
    cam_info,               # dict from camera_info_from_frame()
    weight_thresh=0.3,
):
    """
    Backproject keypoints (u,v) using expected depth map (Dep/Wei) and weight map (Wei).
    cam_info:
      fx, fy, cx, cy, rotation (R_c2w 3x3), position (t_c2w 3,)
    depth_hw/weight_hw: (H,W)
    kpts_uv: (N,2) float pixel coords (subpixel)
    
    depth_hw is expected camera-space Z (weighted by alpha*T)

    """

    kpts_uv = np.asarray(kpts_uv, dtype=np.float32).reshape(-1, 2)
    depth_hw = np.asarray(depth_hw)
    weight_hw = np.asarray(weight_hw)

    
    assert depth_hw.ndim == 2
    assert weight_hw.shape == depth_hw.shape

    H, W = depth_hw.shape

    fx = float(cam_info["fx"])
    fy = float(cam_info["fy"])
    cx = float(cam_info["cx"])
    cy = float(cam_info["cy"])

    R_c2w = np.asarray(cam_info["rotation"], dtype=np.float32).reshape(3, 3)
    t_c2w = np.asarray(cam_info["position"], dtype=np.float32).reshape(3,)

    # --- subpixel coords ---
    u = kpts_uv[:, 0]
    v = kpts_uv[:, 1]
    
    # inside image
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(inside):
        Xw_empty = np.zeros((0, 3), np.float32)
        empty_idx = np.zeros((0,), np.int64)
        return Xw_empty, empty_idx
    idx_in = np.nonzero(inside)[0]
    u_in = u[inside]
    v_in = v[inside]

    # # bilinear sampling
    # z = bilinear_sample_hw(depth_hw, u_in, v_in).astype(np.float32)
    # w = bilinear_sample_hw(weight_hw, u_in, v_in).astype(np.float32)

    # valid = np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w >= weight_thresh)
    # if not np.any(valid):
    #     Xw_empty = np.zeros((0, 3), np.float32)
    #     empty_idx = np.zeros((0,), np.int64)
    #     return Xw_empty, empty_idx

    # valid_idx = idx_in[valid].astype(np.int64)
    # u_k = u_in[valid].astype(np.float32)
    # v_k = v_in[valid].astype(np.float32)
    # z_k = z[valid]
    
    acc_depth = bilinear_sample_hw(depth_hw,  u_in, v_in).astype(np.float32)
    acc_w     = bilinear_sample_hw(weight_hw, u_in, v_in).astype(np.float32)

    valid = np.isfinite(acc_depth) & np.isfinite(acc_w) & (acc_w >= weight_thresh)
    zmax = float(cam_info.get("zfar", np.inf))
    z = acc_depth[valid] / (acc_w[valid] + 1e-8)      
    valid2 = np.isfinite(z) & (z > 0.0)& (z < zmax) 

    # 二次筛选
    valid_idx = idx_in[valid][valid2]
    u_k = u_in[valid][valid2]
    v_k = v_in[valid][valid2]
    z_k = z[valid2]

    # backproject (camera Z)
    x = (u_k - cx) / fx * z_k
    y = (v_k - cy) / fy * z_k
    Xc = np.stack([x, y, z_k], axis=1).astype(np.float32)

    # camera -> world
    Xw = (R_c2w @ Xc.T).T + t_c2w[None, :]
    

    return Xw.astype(np.float32), valid_idx

 
 
def get_neighbor_vids(vid_mid: int, num_views: int, neighbor_delta:int=1, circular: bool = True):
    nbs = []
    for d in range(1, int(neighbor_delta) + 1):
        if circular:
            nbs.append((vid_mid - d) % num_views)
            nbs.append((vid_mid + d) % num_views)
        else:
            if vid_mid - d >= 0: nbs.append(vid_mid - d)
            if vid_mid + d < num_views: nbs.append(vid_mid + d)
    # 去重保持顺序
    seen = set()
    out = []
    for x in nbs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out
    
def project_world_to_uvz(Xw: np.ndarray, cam_info: dict):
    Xw = np.asarray(Xw, dtype=np.float32).reshape(-1, 3)

    R_c2w = np.asarray(cam_info["rotation"], dtype=np.float32).reshape(3, 3)
    t_c2w = np.asarray(cam_info["position"], dtype=np.float32).reshape(3,)
    R_w2c = R_c2w.T

    Xc = (R_w2c @ (Xw - t_c2w[None, :]).T).T  # (N,3)
    z = Xc[:, 2].astype(np.float32)

    fx = float(cam_info["fx"]); fy = float(cam_info["fy"])
    cx = float(cam_info["cx"]); cy = float(cam_info["cy"])

    eps = 1e-8
    u = fx * (Xc[:, 0] / (z + eps)) + cx
    v = fy * (Xc[:, 1] / (z + eps)) + cy
    return u.astype(np.float32), v.astype(np.float32), z

def build_keypoint_gaussians(
    Xw: np.ndarray,           # (M,3)
    device,
    scale=0.002,
    opacity=1.0,
    color=(1.0, 0.0, 0.0),    # red
):
    """
    Return a GaussianModel containing only keypoints.
    """

    M = Xw.shape[0]
    gm = GaussianModel()

    gm.means3D = torch.from_numpy(Xw).float().to(device)
    gm.means2D = torch.zeros((M, 2), device=device)  # dummy

    gm.scales = torch.full((M, 3), scale, device=device)
    gm.rotations = torch.zeros((M, 4), device=device)
    gm.rotations[:, 0] = 1.0  # identity quaternion

    gm.opacities = torch.full((M, 1), opacity, device=device)
    gm.colors_precomp = (
        torch.tensor(color, device=device)
        .view(1, 3, 1)
        .repeat(M, 1, 1)
    )

    return gm

def multiview_keep_mask(
    Xw: np.ndarray,
    neighbor_cam_infos: list,
    neighbor_depths: list,
    neighbor_weights: list,
    neighbor_delta: int,
    weight_thresh: float = 0.3,
    tau_rel: float = 0.02,
    tau_abs: float = 0.01,
):
    """
    keep[i] = True  iff  point i passes consistency check in >= neighbor_delta views
    """
    Xw = np.asarray(Xw, np.float32).reshape(-1, 3)
    N = Xw.shape[0]
    support_count = np.zeros((N,), dtype=np.int32)

    for cam_j, D_j, W_j in zip(neighbor_cam_infos, neighbor_depths, neighbor_weights):
        u, v, z_pred = project_world_to_uvz(Xw, cam_j)
        H, W = D_j.shape

        inside = (
            (u >= 0) & (u <= W - 1) &
            (v >= 0) & (v <= H - 1) &
            np.isfinite(z_pred) & (z_pred > 1e-6)
        )
        if not np.any(inside):
            continue

        idx = np.nonzero(inside)[0]
        u_in, v_in, z_in = u[inside], v[inside], z_pred[inside]

        z_r = bilinear_sample_hw(D_j, u_in, v_in)
        w_r = bilinear_sample_hw(W_j, u_in, v_in)

        ok = (
            (w_r >= weight_thresh) &
            np.isfinite(z_r) & (z_r > 0) &
            (np.abs(z_r - z_in) < np.maximum(tau_abs, tau_rel * z_in))
        )

        support_count[idx[ok]] += 1

    keep = support_count >= int(neighbor_delta)
    return keep

    
def select_geometric_neighbors_from_cam_infos(cam_infos_all, vid_mid, M=8, w_pos=1.0, w_ang=0.5):
    Cmid = np.asarray(cam_infos_all[vid_mid]["position"], np.float32)
    Rmid = np.asarray(cam_infos_all[vid_mid]["rotation"], np.float32).reshape(3,3)

    # forward in world: try +Z; if wrong, change to -Z
    fmid = Rmid @ np.array([0,0,1], np.float32)
    fmid = fmid / (np.linalg.norm(fmid) + 1e-8)

    scores = []
    for j, camj in enumerate(cam_infos_all):
        if j == vid_mid:
            continue
        Cj = np.asarray(camj["position"], np.float32)
        Rj = np.asarray(camj["rotation"], np.float32).reshape(3,3)
        fj = Rj @ np.array([0,0,1], np.float32)
        fj = fj / (np.linalg.norm(fj) + 1e-8)

        dpos = float(np.linalg.norm(Cj - Cmid))
        cosang = float(np.clip(np.dot(fj, fmid), -1.0, 1.0))
        dang = float(np.arccos(cosang))   # radians

        score = w_pos * dpos + w_ang * dang
        scores.append((score, j))

    scores.sort(key=lambda x: x[0])
    return [j for _, j in scores[:M]]

def debug_check_reprojection(kpt_uv, depth_hw, weight_hw, cam_info, weight_thresh=0.3, max_show=2000):
    """
    1) 用 backproject_kpts_to_world 得到 Xw
    2) 再用 project_world_to_uvz 投回 uv
    3) 看 reprojection error (px)
    """
    Xw, idx = backproject_kpts_to_world(
        kpts_uv=kpt_uv,
        depth_hw=depth_hw,
        weight_hw=weight_hw,
        cam_info=cam_info,
        weight_thresh=weight_thresh,
    )
    if len(Xw) == 0:
        print("[DBG] no valid backproject points")
        return

    u2, v2, z2 = project_world_to_uvz(Xw, cam_info)
    uv2 = np.stack([u2, v2], axis=1)

    uv_ref = kpt_uv[idx]  # 原始 keypoints 的对应子集

    # reprojection error in pixels
    err = np.linalg.norm(uv2 - uv_ref, axis=1)

    print(
        f"[DBG] reproj_err px: "
        f"median={np.median(err):.3f}  "
        f"p90={np.percentile(err,90):.3f}  "
        f"p95={np.percentile(err,95):.3f}  "
        f"max={err.max():.3f}  "
        f"valid={len(err)}/{len(kpt_uv)}"
    )

    # 可视化：随机抽一些点画回图
    if max_show is not None and max_show > 0:
        n = min(max_show, len(err))
        sel = np.random.choice(len(err), size=n, replace=False)

        plt.figure(figsize=(6,6))
        plt.title("Reprojection check (blue=orig kpt, red=reprojected)")
        # 背景可选：你可以传入 RGB 图像来 show，这里只画散点
        plt.scatter(uv_ref[sel,0], uv_ref[sel,1], s=8, c="deepskyblue", linewidths=0, label="orig")
        plt.scatter(uv2[sel,0],    uv2[sel,1],    s=8, c="red",         linewidths=0, label="reproj")
        plt.legend()
        plt.gca().invert_yaxis()  # 如果你的图像坐标 y 向下，这行可开/关试一下
        plt.show()

def debug_nn_distance(
    Xw: np.ndarray,          # (M,3) backprojected world points
    gs_xyz: np.ndarray,      # (N,3) Gaussian means (world)
    max_xw: int = 200,       # subsample Xw
    max_gs: int = 5000,      # subsample GS points
    seed: int = 0,
    tag: str = "",
):
    """
    Compute NN distance from Xw to GS surface (brute force, subsampled).
    """
    if len(Xw) == 0:
        print(f"[DBG][{tag}] Xw empty")
        return

    rng = np.random.default_rng(seed)

    Xw = np.asarray(Xw, np.float32)
    gs_xyz = np.asarray(gs_xyz, np.float32)

    # subsample for speed
    if len(Xw) > max_xw:
        Xw_s = Xw[rng.choice(len(Xw), size=max_xw, replace=False)]
    else:
        Xw_s = Xw

    if len(gs_xyz) > max_gs:
        gs_s = gs_xyz[rng.choice(len(gs_xyz), size=max_gs, replace=False)]
    else:
        gs_s = gs_xyz

    # brute-force NN
    # d2: (Mx, Ng)
    d2 = ((Xw_s[:, None, :] - gs_s[None, :, :]) ** 2).sum(axis=2)
    nn = np.sqrt(d2.min(axis=1))

    print(
        f"[DBG][{tag}] Xw->GS NN dist: "
        f"mean={nn.mean():.5f}  "
        f"median={np.median(nn):.5f}  "
        f"p95={np.percentile(nn,95):.5f}  "
        f"max={nn.max():.5f}  "
        f"(Xw_s={len(Xw_s)}, GS_s={len(gs_s)})"
    )

    return nn

# ----------------------------
# Main
# ----------------------------
def main(shape_splat_root: str,
         list_txt: str,
         out_root= None,
         K: int = 3,
         base_seed: int = 0,
         top_k: int = 1024,
         device: str = "cuda",
         width: int = 400,
         height: int = 400,
         split_name: str = "train",
         overwrite: bool = False,
         debug: bool = False,
         covisible: bool = False,):

    
    transforms_json_path = os.path.join(shape_splat_root, "transforms.json")
    assert os.path.exists(transforms_json_path), f"Missing: {transforms_json_path}"

    with open(transforms_json_path, "r") as f:
        tf = json.load(f)
    frames = tf["frames"]
    camera_angle_x = tf["camera_angle_x"]
    num_views = len(frames)
    assert num_views ==72

    ply_list = read_list(list_txt)
    print(f"[INFO] Loaded {len(ply_list)} scenes from {list_txt}")

    xfeat = load_xfeat(device=device)
    cam_infos_all = [
    camera_info_from_frame(frames[i], camera_angle_x, width=width, height=height)
    for i in range(num_views)]

    for idx, ply_name in enumerate(ply_list):
        model_path, category = resolve_model_path(shape_splat_root, ply_name)
        if model_path is None:
            print(f"[MISS] cannot find: {ply_name}")
            continue

        scene_id = os.path.splitext(os.path.basename(model_path))[0]
        if out_root is not None:
            out_path = os.path.join(out_root, category, f"{scene_id}.h5")

            if (not overwrite) and os.path.exists(out_path):
                print(f"[SKIP] exists: {out_path}")
                continue


        # per-scene random views (different for different scenes; reproducible)
        sseed = stable_scene_seed(base_seed, scene_id)
        view_indices = sample_view_indices(num_views, K, seed=sseed)

        model = GaussianModel()
        model.load(model_path)
        
        
        # ---- DEBUG: gaussian world stats ----
        xyz_gs = model.means3D.detach().cpu().numpy()   # (N,3)
        cent_gs = xyz_gs.mean(axis=0)
        rad_gs = np.max(np.linalg.norm(xyz_gs - cent_gs[None, :], axis=1))
        print(f"[DBG][{scene_id}] gs mean={cent_gs} rad={rad_gs} N={xyz_gs.shape[0]}")
        
        
        imgs, cam_infos, depths, weights = [], [], [], []

        for vid in view_indices:
            frame = frames[int(vid)]
            cam_info = camera_info_from_frame(frame, camera_angle_x, width=width, height=height)
            with suppress_stdout_stderr():
                rgb, depth, weight = render_one_rgbdw(model, cam_info)
            if rgb.min() == rgb.max() == 0:
                raise AssertionError(f"[EMPTY RENDER] scene={scene_id} view={int(vid)} img_minmax=(0,0)")
            imgs.append(rgb)
            cam_infos.append(cam_info)
            depths.append(depth)
            weights.append(weight)

            
                
        xfeat_results = run_xfeat(xfeat, imgs, top_k=top_k, device=device)  
        
        
        # ---neighbor and covisible check ---
        weight_thresh = 0.5
        tau_rel = 0.02
        tau_abs = 0.01
        final_results = []

        for v, vid_mid in enumerate(view_indices):
            vid_mid = int(vid_mid)

            # --- XFeat raw ---
            kpt_uv = np.asarray(xfeat_results[v]["kpt"], dtype=np.float32)
            feat   = np.asarray(xfeat_results[v]["feat"], dtype=np.float32)
            conf   = np.asarray(xfeat_results[v]["conf"], dtype=np.float32)

            # if debug:
                # show_render_and_xfeat(
                #     rgb=imgs[v],
                #     kpt_uv=kpt_uv,
                #     conf=conf,
                #     title=f"{scene_id} | file_path={vid} | kpt={len(kpt_uv)}"
                # )       
            # (1) reference backproject (weight gate)

            Xw, idx_ref = backproject_kpts_to_world(
                kpts_uv=kpt_uv,
                depth_hw=depths[v],
                weight_hw=weights[v],
                cam_info=cam_infos[v],
                weight_thresh=weight_thresh,
            )
            if debug:
                print(f"[STAT] weight_th={weight_thresh}  valid3D={len(Xw)}/{len(kpt_uv)}")
                # debug_nn_distance(
                #     Xw=Xw,
                #     gs_xyz=xyz_gs,
                #     tag=f"{scene_id}|vid{vid_mid}"
                # )

            if Xw.shape[0] == 0:
                D = feat.shape[1] if feat.ndim == 2 else 0
                final_results.append({
                    "view_index": vid_mid,
                    "kptuv": kpt_uv,
                    "feat": feat,
                    "conf": conf,
                    "kptxyz": np.zeros((0,3), np.float32),
                    "xyz_idx": np.zeros((0,), np.int64),
                })
                continue

            Xw_surf, idx_surf, nn_dist = nn_filter_and_snap_ckdtree(
                Xw, xyz_gs, idx_ref, dist_thresh=0.6, snap=True
            )

            xyz_final = Xw_surf
            if 'train' in split_name:

                
                # print("nn_dist med/p90/max:", np.median(nn_dist), np.percentile(nn_dist,90), np.max(nn_dist))
                kpt_uv_f = kpt_uv[idx_surf] 
                feat_f = feat[idx_surf] 
                conf_f = conf[idx_surf] 
                xyz_idx_final = np.arange(len(kpt_uv_f), dtype=np.int64)

                assert xyz_final.shape[0] == kpt_uv_f.shape[0] == feat_f.shape[0] == conf_f.shape[0]== xyz_idx_final.shape[0]
            else:
                kpt_uv_f = kpt_uv
                feat_f = feat
                conf_f = conf
                xyz_idx_final = idx_surf
            if debug:
                show_render_and_xfeat( imgs[v], imgs[v], kpt_uv, kpt_uv[idx_surf], title=f"{scene_id} | cat={category}| {len(kpt_uv)} -> {len(kpt_uv[idx_surf])}")
                
                        
            final_results.append({
                "view_index": vid_mid,
                "kptuv": kpt_uv_f,
                "feat": feat_f,
                "conf": conf_f,
                "kptxyz": xyz_final,
                "xyz_idx": xyz_idx_final,
            })
            # final_results.append({
            #     "view_index": vid_mid,
            #     "kptuv": kpt_uv,
            #     "feat": feat,
            #     "conf": conf,
            #     "kptxyz": xyz_final,
            #     "xyz_idx": xyz_idx_final,
            # })
            
        if out_root is not None:
            saved = save_scene_h5(
                out_root=out_root,
                category=category,
                scene_id=scene_id,
                model_path=model_path,
                view_indices=view_indices,
                cam_infos=cam_infos,
                results=final_results,          # 注意这里用 final_results
                split_name=split_name,
            )

            nk_all = [r["kptuv"].shape[0] for r in final_results]
            nk_xyz = [r["kptxyz"].shape[0] for r in final_results]
            print(f"[OK {idx+1}/{len(ply_list)}] {category}/{scene_id} views={view_indices.tolist()} "
                f"n_kptuv={nk_all} n_kptxyz={nk_xyz} -> {saved}")
        
        
        

def parse_cat_from_ply(ply_name: str) -> str:
    return os.path.basename(ply_name).split("-", 1)[0]

def parse_test_token_from_ply(ply_name: str, ndigits=6):
    stem = os.path.splitext(os.path.basename(ply_name))[0]
    m = re.search(r"test(\d+)$", stem)
    return m.group(1).zfill(ndigits) if m else None

def is_train_ply(ply_name: str):
    return "train" in os.path.basename(ply_name)

def is_test_ply(ply_name: str):
    return "test" in os.path.basename(ply_name)


def nn_filter_and_snap_ckdtree(Xw, xyz_gs, valid_idx, dist_thresh=1.0, snap=True):
    """
    Xw: (M,3) backprojected points
    xyz_gs: (N,3) gaussian centers
    valid_idx: (M,) indices mapping to original keypoints
    dist_thresh: keep points whose NN distance < dist_thresh
    snap: if True, return snapped points on GS centers; else return filtered original Xw
    """
    Xw = np.asarray(Xw, np.float32)
    xyz_gs = np.asarray(xyz_gs, np.float32)
    valid_idx = np.asarray(valid_idx)

    tree = cKDTree(xyz_gs)
    nn_dist, nn_idx = tree.query(Xw, k=1, workers=-1)  # nn_dist: (M,), nn_idx: (M,)

    keep = nn_dist < dist_thresh

    if snap:
        X_out = xyz_gs[nn_idx[keep]]
    else:
        X_out = Xw[keep]

    return X_out, valid_idx[keep], nn_dist[keep]


def build_img_dict(cat_dir: str):
    png_files = sorted(glob.glob(os.path.join(cat_dir, "*.png")))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in png_files}


def show_render_and_xfeat(rgb_img, rgb_rendered,  kpt_uv, kpt_uv_f, conf=None, title=""):
    """
    Show XFeat keypoints on:
      - left: rendered RGB
      - right: real image RGB
    Using the SAME uv coordinates.
    """

    # ---------- prepare images ----------
    def _prep(img):
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
        return img

    img_r = _prep(rgb_rendered)
    img_i = _prep(rgb_img)

    H, W = img_r.shape[:2]
    uv = np.asarray(kpt_uv)
    uv_f = np.asarray(kpt_uv_f)

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    # ---- left: real image ----
    axes[0].imshow(img_i)
    axes[0].scatter(uv[:, 0], uv[:, 1], s=6, c="red", linewidths=0)
    axes[0].set_title("Raw Detection")
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)
    axes[0].axis("off")
    
    # ---- right: rendered ----
    axes[1].imshow(img_r)
    axes[1].scatter(uv_f[:, 0], uv_f[:, 1], s=6, c="red", linewidths=0)
    axes[1].set_title("Align to surface")
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].axis("off")

    # ---------- title ----------
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

    # # ---------- optional confidence histogram ----------
    # if conf is not None and len(conf) > 0:
    #     plt.figure(figsize=(6, 3))
    #     plt.hist(np.asarray(conf), bins=30)
    #     plt.title("XFeat confidence histogram")
    #     plt.show()

        
def main_scared(
    list_txt:str,
    splat_root: str,              # ROOT containing cat folders
    cam_root:str,
    out_root: str,
    gt_img_root:str,
    device: str = "cuda",
    top_k: int = 1024,
    overwrite: bool = False,
    debug: bool = False,
    split_name: str = "scared",

):

    # -------------------------
    # 1) read txt and group by cat
    # -------------------------
    ply_list = read_list(list_txt)
    print(f"[INFO] Loaded {len(ply_list)} scenes from {list_txt}")
    xfeat = load_xfeat(device=device)


      
    bad_list=[]
    for idx, ply_name in enumerate(ply_list): # d1k1-test000010.ply
        scene_id = os.path.splitext(os.path.basename(ply_name))[0] # d1k1-test000010
        prefix, rest = scene_id.split("-", 1)   # prefix=d1k1, rest=test000010
        cat = prefix

        
        # load gs, camera info
        model_path  = os.path.join(splat_root, cat, f"{scene_id}.ply") 
        cam_info_path  = os.path.join(cam_root, cat, f"{scene_id}.json") 
        # gt image
        # 提取数字部分 (例如从 "d5k1" 中提取 "5")
        digit_match = re.match(r'd(\d+)', prefix).group(1)

        # 构建正确的文件夹名称: dataset_5
        dataset = f"dataset_{digit_match}"
        split = re.match(r'(train|test)', rest).group(1)
        # frame index -> 5 digits
        frame_id = re.search(r'\d+', rest).group()
        frame_id = f"{int(frame_id):05d}"
        # build path
        gt_img_path = os.path.join(
            gt_img_root,
            dataset,
            cat,
            split,
            "ours_30000",
            "gt",
            f"{frame_id}.png"
        )
        with open(cam_info_path, "r") as f:
            cam_info = json.load(f)

        
        # out root
        if out_root is not None:
            out_path = os.path.join(out_root, cat, f"{scene_id}.h5")

            if (not overwrite) and os.path.exists(out_path):
                print(f"[SKIP] exists: {out_path}")
                continue

        model = GaussianModel()
        model.load(model_path)

        # ---- DEBUG: gaussian world stats ----
        xyz_gs = model.means3D.detach().cpu().numpy()   # (N,3)
        cent_gs = xyz_gs.mean(axis=0)
        rad_gs = np.max(np.linalg.norm(xyz_gs - cent_gs[None, :], axis=1))
        # print(f"[DBG][{scene_id}] gs mean={cent_gs} rad={rad_gs} N={xyz_gs.shape[0]}")
        # with suppress_stdout_stderr():
        rgb, depth, weight = render_one_rgbdw(model, cam_info) 
        
        
        # 读取图像并转换为 RGB (OpenCV 默认是 BGR)
        gt_img = cv2.imread(gt_img_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)  
                    
        xres = run_xfeat(xfeat, [gt_img], top_k=top_k, device=device) 
        final_results = []   

        kpt_uv, feat, conf = xres[0]["kpt"], xres[0]["feat"], xres[0]["conf"]

        
        # backproject
        Xw_raw, idx_ref = backproject_kpts_to_world(
            kpts_uv=kpt_uv,
            depth_hw=depth,
            weight_hw=weight,
            cam_info=cam_info,
            weight_thresh=0.05,
        )

 
        Xw_surf, idx_surf, nn_dist = nn_filter_and_snap_ckdtree(
            Xw_raw, xyz_gs, idx_ref, dist_thresh=0.6, snap=True
        )
        
                
        
        
        
        # print("nn_dist med/p90/max:", np.median(nn_dist), np.percentile(nn_dist,90), np.max(nn_dist))
        kpt_uv_f = kpt_uv[idx_surf] 
        feat_f = feat[idx_surf] 
        conf_f = conf[idx_surf] 

            

        # # ---- bad sample filter (localization-oriented) ----
        # min_pairs = 200
        # p90_thresh = 1.0      # mm
        # keep_ratio_min = 0.5  # optional

        # n0 = len(kpt_uv)
        # n1 = len(kpt_uv_f)
        # keep_ratio = (n1 / max(n0, 1))

        # p90 = float(np.percentile(nn_dist, 90)) if len(nn_dist) else float("inf")

        # is_bad = (n1 < min_pairs) or (p90 > p90_thresh) or (keep_ratio < keep_ratio_min)

        # if is_bad:
        #     # 你想要的格式: d*k*-000008.ply
        #     # scene_id 类似: d5k1-1-000008  (你现在确实是这个格式)
        #     bad_list.append(f"{scene_id}.ply")
        #     print(f"[BAD] {scene_id}  n={n1}/{n0} keep={keep_ratio:.2f}  p90={p90:.3f}mm")
        #     continue  # 直接跳过保存 h5（可选）        
        
                    
        if debug:
            show_render_and_xfeat( gt_img, rgb, kpt_uv, kpt_uv_f, title=f"{scene_id} | cat={cat}| {len(kpt_uv)} -> {len(kpt_uv_f)}")
            
        xyz_final = Xw_surf
        if 'train' in split_name:
            # print("nn_dist med/p90/max:", np.median(nn_dist), np.percentile(nn_dist,90), np.max(nn_dist))
            kpt_uv_f = kpt_uv[idx_surf] 
            feat_f = feat[idx_surf] 
            conf_f = conf[idx_surf] 
            xyz_idx_final = np.arange(len(kpt_uv_f), dtype=np.int64)

            assert xyz_final.shape[0] == kpt_uv_f.shape[0] == feat_f.shape[0] == conf_f.shape[0]== xyz_idx_final.shape[0]
        else:
            kpt_uv_f = kpt_uv
            feat_f = feat
            conf_f = conf
            xyz_idx_final = idx_surf
                    
        final_results.append({
            "view_index": idx,
            "kptuv": kpt_uv_f,
            "feat": feat_f,
            "conf": conf_f,
            "kptxyz": xyz_final,
            "xyz_idx": xyz_idx_final,
        })

        if out_root is not None:
            saved = save_scene_h5(
                out_root=out_root,
                category=cat,
                scene_id=scene_id,
                model_path=model_path,
                view_indices=[0],
                cam_infos=[cam_info],
                results=final_results,          # 注意这里用 final_results
                split_name=split_name,
            )
            
            # print(f"[OK] {saved}")
            print(f"n_kptuv={len(kpt_uv)} n_kptxyz={len(kpt_uv_f)} -> {saved}")
    # bad_path = os.path.join(out_root, "bad_sample.txt")
    # with open(bad_path, "w") as f:
    #     for s in sorted(set(bad_list)):
    #         f.write(s + "\n")
    # print(f"[DONE] wrote bad_sample.txt to: {bad_path}  (N={len(set(bad_list))})")    


if __name__ == "__main__":
    run_original = True


    if run_original:
        shape_splat_root = "/mnt/nct-zfs/TCO-All/SharedDatasets/ShapeSplat" 
        split_name = "scene_train"   # "train" or "val", "test"
        list_txt = f"/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/cfg/split/{split_name}.txt"
        out_root = "/mnt/nct-zfs/TCO-All/SharedDatasets/ShapeSplat/XFeat_upgrade" 
        main(
            shape_splat_root=shape_splat_root,
            list_txt=list_txt,
            out_root=None,
            K=3,
            base_seed=0,
            top_k=1024,
            device="cuda",
            width=400,
            height=400,
            split_name=split_name,
            overwrite=True,
            debug=True,
            covisible = False
        )
    else:
        shape_splat_root = "/mnt/nct-zfs/TCO-All/Projects/MisGS/liwen_rp_output/scared_splat" 
        split_name = "scared_test"   # "train" or "val", "test"
        # list_txt = f"/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/ext/EndoGaussian/scared_full.txt"
        list_txt = f"/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/cfg/split/{split_name}.txt"
        out_root = f"/mnt/nct-zfs/TCO-All/Projects/MisGS/liwen_rp_output/scared_XFeat" 
        # out_root = None
        gt_img_root = '/mnt/cluster/workspaces/students/liwenliu/2025_Master_Thesis/localdata/Scared_stereo'
        cam_root  = f'/mnt/nct-zfs/TCO-All/Projects/MisGS/liwen_rp_output/scared_cam'
        main_scared(
            list_txt=list_txt,
            splat_root= shape_splat_root,
            cam_root=cam_root,
            out_root=out_root,
            gt_img_root=gt_img_root,
            top_k=1024,
            device="cuda",
            split_name=split_name,
            overwrite=False,
            debug=False,)
        
