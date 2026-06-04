import random
import numbers
from PIL import Image, ImageMath
import os
import os.path
import numpy as np
import struct
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pathlib import Path

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_pc(pc_np, z_cutoff=1000, birds_view=False, color='height', size=0.3, ax=None, cmap=cm.jet, is_equal_axes=True, mins = None, maxs = None):
    # remove large z points
    # valid_index = pc_np[:, ] < z_cutoff
    # pc_np = pc_np[valid_index, :]

    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d') 
    if type(color)==str and color == 'height':
        # c = pc_np[:, 0]
        c = 'grey'
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=size, c=c, cmap=cmap, edgecolors='none')
    elif type(color)==str and color == 'reflectance':
        assert False
    elif type(color) == np.ndarray:
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=size, c=color, cmap=cmap, edgecolors='none')
    else:
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=size, c=color, edgecolors='none')

    if is_equal_axes:
        axisEqual3D(ax)
    ax.view_init(elev=20, azim=30)

    if mins is not None and maxs is not None:
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
    # if True == birds_view:
    #     ax.view_init(elev=0, azim=-90)
    # else:
    #     ax.view_init(elev=-45, azim=-90)
    # ax.invert_yaxis()

    return ax


def plot_pc_old(pc_np, z_cutoff=70, birds_view=False, color='height', size=0.3, ax=None):
    # remove large z points
    valid_index = pc_np[:, 2] < z_cutoff
    pc_np = pc_np[valid_index, :]

    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = Axes3D(fig)
    if color == 'height':
        c = pc_np[:, 1]
        ax.scatter(pc_np[:, 0].tolist(), pc_np[:, 1].tolist(), pc_np[:, 2].tolist(), s=size, c=c, cmap=cm.jet_r)
    elif color == 'reflectance':
        assert False
    else:
        ax.scatter(pc_np[:, 0].tolist(), pc_np[:, 1].tolist(), pc_np[:, 2].tolist(), s=size, c=color)

    axisEqual3D(ax)
    if True == birds_view:
        ax.view_init(elev=0, azim=-90)
    else:
        ax.view_init(elev=-45, azim=-90)
    # ax.invert_yaxis()

    return ax




# -----------Cage-------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
def plot_keypoint_correspondence(src_kpt, dst_kpt, sample_idx=0, max_points=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    """
    src_kpt, dst_kpt: Tensor, shape (B, 3, K) or (3, K)
    sample_idx: which batch to visualize
    max_points: optional, only show first N points for clarity
    """
    if src_kpt.ndim == 3:
        src = src_kpt[sample_idx].detach().cpu().numpy()
        dst = dst_kpt[sample_idx].detach().cpu().numpy()
    else:
        src = src_kpt.detach().cpu().numpy()
        dst = dst_kpt.detach().cpu().numpy()

    # src, dst shape: (3, K)
    K = src.shape[1]
    if max_points is not None:
        K = min(K, max_points)
        src, dst = src[:, :K], dst[:, :K]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Keypoint Correspondence (red=src, blue=dst)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # scatter
    ax.scatter(src[0], src[1], src[2], color='red', s=35, alpha=0.9, zorder=3)
    ax.scatter(dst[0], dst[1], dst[2], color='blue', s=20, alpha=0.6, zorder=2)


    # draw lines between corresponding indices
    for i in range(K):
        ax.plot([src[0, i], dst[0, i]],
                [src[1, i], dst[1, i]],
                [src[2, i], dst[2, i]],
                color='gray', alpha=0.6, linewidth=1)

    # annotate first few indices
    for i in range(min(K, 20)):
        ax.text(dst[0, i], dst[1, i], dst[2, i], str(i), color='black', fontsize=8)

    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_cage(cage, points, faces, save_path: Path = None, title=None, b= 0):
    """
    cage:   (B,3,V)
    points: (B,3,N)
    faces:  (1,F,3) or (B,F,3)
    save_path: Path 对象，如果提供则保存图片
    """
    
    cage_b = cage[b]
    if cage_b.shape[0] == 3:
        verts = cage_b.detach().cpu().numpy().T  # (V,3)
    else:
        verts = cage_b.detach().cpu().numpy()
        
    pts = points[b].detach().cpu().numpy().T



    # faces: (1,F,3) or (B,F,3) -> (F,3) int
    faces = faces[0].detach().cpu().numpy()
    if faces.ndim == 3:
        faces = faces[b]

    # ---- build unique edge segments from triangle faces ----
    # each face (i,j,k) -> edges (i,j),(j,k),(k,i)
    tri_edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)
    # sort each edge's endpoints so (i,j) == (j,i), then unique
    tri_edges = np.sort(tri_edges, axis=1)
    edges = np.unique(tri_edges, axis=0)  # (E,2)

    # segments: list of 2-point arrays shaped (2,3)
    segs = np.stack([verts[edges[:,0]], verts[edges[:,1]]], axis=1)  # (E,2,3)

    # ---- plot ----
    fig = plt.figure(figsize=(6,6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # point cloud (semi-transparent blue)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
            s=1, alpha=0.35, c="#4da3ff", label="points")

    # cage wireframe (thin gray lines)
    lc = Line3DCollection(segs, linewidths=0.8, colors="#b0b7bf", alpha=1.0)
    ax.add_collection3d(lc)

    # --- set equal axes so it looks like a tight cage ---
    allp = np.concatenate([pts, verts], axis=0)
    mn, mx = allp.min(0), allp.max(0)
    ctr = (mn + mx) / 2
    rad = (mx - mn).max() * 0.55 + 1e-9
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.view_init(elev=25, azim=35)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)
    else:
        plt.show()

from visdom import Visdom
def plot_cage_visdom(vis: Visdom, cage, points, faces, title=None, env='main', win=None):
    """
    在 Visdom 中同时画点云 + 笼子线框。
    支持输入为 numpy.ndarray 或 torch.Tensor。
    要求:
      - cage:   (V,3) or (3,V)
      - points: (N,3) or (3,N)
      - faces:  (F,3) 或 (1,F,3)
    """
    import numpy as np
    import torch

    # ---- convert inputs to numpy safely ----
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    cage = to_numpy(cage)
    points = to_numpy(points)
    faces = to_numpy(faces)

    # ---- reshape cage/points ----
    if cage.shape[0] == 3:  # (3,V)
        verts = cage.T
    else:                   # (V,3)
        verts = cage

    if points.shape[0] == 3:
        pts = points.T
    else:
        pts = points

    # ---- faces ----
    if faces.ndim == 3:
        faces = faces[0]
    faces = faces.astype(np.int32)

    # ---- build edges from triangle faces ----
    tri_edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)
    tri_edges = np.sort(tri_edges, axis=1)
    edges = np.unique(tri_edges, axis=0)  # (E,2)

    seg = verts[edges]  # (E,2,3)

    # ---- convert everything to Python lists for Visdom ----
    xs, ys, zs = [], [], []
    for a, b in seg:
        xs += [float(a[0]), float(b[0]), None]
        ys += [float(a[1]), float(b[1]), None]
        zs += [float(a[2]), float(b[2]), None]

    x_pts = pts[:, 0].astype(float).tolist()
    y_pts = pts[:, 1].astype(float).tolist()
    z_pts = pts[:, 2].astype(float).tolist()

    # ---- prepare Visdom data ----
    win = None if win is None else str(win)
    env = str(env)
    title = str(title) if title is not None else 'Cage over points'

    traces = [
        dict(
            type='scatter3d',
            x=x_pts, y=y_pts, z=z_pts,
            mode='markers',
            marker=dict(size=2, opacity=0.6),
            name='points'
        ),
        dict(
            type='scatter3d',
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(width=2),
            opacity=1.0,
            name='cage (wire)'
        ),
    ]

    layout = dict(
        title=title,
        scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z'),
            aspectmode='data'
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    msg = dict(data=traces, layout=layout, win=win, eid=env)
    resp = vis._send(msg)
    return resp['win'] if isinstance(resp, dict) and 'win' in resp else win
