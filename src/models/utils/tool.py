import math
import torch

def apply_transformation(points, R, s, t):
    """
    Apply SIM(3) transform to a batch of point sets.

    Args:
        points: (B, 3, N)    # e.g., cage or keypoints
        R:      (B, 3, 3)    # rotation
        s:      scalar or (B,) or (B,1,1)  # scale
        t:      (B, 3, 1)    # translation

    Returns:
        (B, 3, N) transformed points
    """
    # Ensure s is a tensor of shape (B,1,1) on the right device/dtype
    if not torch.is_tensor(s):
        s = torch.tensor(s, dtype=points.dtype, device=points.device)
    if s.dim() == 0:
        s = s.expand(points.shape[0])
    if s.dim() == 1:
        s = s.view(-1, 1, 1)  # (B,1,1)

    # Make sure R,t are on same device/dtype
    R = R.to(points.device, points.dtype)
    t = t.to(points.device, points.dtype)

    out = torch.matmul(R, points)          # (B,3,3) x (B,3,N) -> (B,3,N)
    out = out * s                          # broadcast (B,1,1)
    out = out + t                          # broadcast (B,3,1)
    return out


def sigmoid_ramp(step, start, end, v0=0.0, v1=1.0):
    if step <= start:
        return v0
    if step >= end:
        return v1
    t = (step - start) / float(end - start)   # 0..1
    # 把 t 映射到 sigmoid 区间，-6..6 足够陡且平滑
    x = (t * 12.0) - 6.0
    s = 1.0 / (1.0 + math.exp(-x))
    return v0 + (v1 - v0) * s

def cosine_ramp(step, start, end, v0=0.0, v1=1.0):
    if step <= start:
        return v0
    if step >= end:
        return v1
    t = (step - start) / (end - start)  # 0..1
    # cosine ease-in-out
    s = 0.5 - 0.5 * math.cos(math.pi * t)
    return v0 + (v1 - v0) * s
