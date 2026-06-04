import torch

@torch.no_grad()
def pcgrad_weights_from_gramian(gramian: torch.Tensor) -> torch.Tensor:
    """
    gramian: (m, m) where gramian[i,j] = <g_i, g_j>
    returns: (m,) weights
    """
    device = gramian.device
    dtype = gramian.dtype

    # torchjd 把计算放 CPU，避免 GPU<->CPU 来回拷贝（这里也照做）
    cpu = torch.device("cpu")
    G = gramian.detach().to(device=cpu, dtype=dtype)

    m = G.shape[0]
    weights = torch.zeros(m, device=cpu, dtype=dtype)

    for i in range(m):
        perm = torch.randperm(m)  # 随机顺序（PCGrad 原论文就有随机性）
        w = torch.zeros(m, device=cpu, dtype=dtype)
        w[i] = 1.0

        for j in perm:
            if j == i:
                continue
            inner = G[j] @ w  # <g_j, g_i^PC>（等价形式）
            if inner < 0.0:
                denom = G[j, j].clamp_min(1e-12)  # 防止除 0
                w[j] -= inner / denom

        weights += w

    return weights.to(device=device, dtype=dtype)


def _get_params(optimizer):
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                params.append(p)
    return params

def _flatten_grads(params):
    flat = []
    for p in params:
        if p.grad is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(p.grad.detach().reshape(-1))
    return torch.cat(flat)

def pcgrad_step_from_losses(optimizer, losses, retain_graph=True):
    """
    optimizer: torch optimizer
    losses: list of scalar tensors
    """
    params = _get_params(optimizer)
    m = len(losses)

    grads_flat = []
    for i, L in enumerate(losses):
        optimizer.zero_grad(set_to_none=True)
        L.backward(retain_graph=(retain_graph or i < m - 1))
        grads_flat.append(_flatten_grads(params))

    # Gramian: (m,m)
    J = torch.stack(grads_flat, dim=0)         # m x P
    gramian = J @ J.T                         # m x m

    w = pcgrad_weights_from_gramian(gramian)   # m

    # 合成梯度：g = sum_i w_i g_i
    g = (w.unsqueeze(1) * J).sum(dim=0)        # P

    # 把合成梯度写回 params
    optimizer.zero_grad(set_to_none=True)
    idx = 0
    for p in params:
        numel = p.numel()
        p.grad = g[idx: idx + numel].view_as(p).clone()
        idx += numel

    optimizer.step()
    return w  # 你想打印每个 loss 的权重时很有用
