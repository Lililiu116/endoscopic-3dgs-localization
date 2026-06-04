import torch
import numpy as np


def _radius_neighbors(batched_points: torch.Tensor,
                      batched_lengths: torch.Tensor,
                      radius: float,
                      max_neighbors: int) -> torch.Tensor:
    """
    建立第 0 層的半徑鄰居矩陣，輸出形狀為 [N_total, K]，其中 K=max_neighbors。
    - batched_points: [N_total, 3]
    - batched_lengths: [B]，本工具預期 B=2（兩個子雲）
    會用歐氏距離做半徑檢索，超過半徑或超過 K 的位置以 shadow 索引填充（N_total）。
    """
    assert batched_points.dim() == 2 and batched_points.size(1) == 3
    assert batched_lengths.dim() == 1 and batched_lengths.numel() == 2

    device = batched_points.device
    N_total = batched_points.size(0)
    # 距離矩陣 [N,N]
    dists = torch.cdist(batched_points, batched_points, p=2)

    # 半徑遮罩（True 表示是鄰居）
    within = dists <= radius
    # 自身可以作為鄰居（與 D3Feat 一致性：允許或不允許均可，這裡允許）

    # 轉為固定 K 個鄰居的索引矩陣，超出者補 shadow
    shadow_index = torch.tensor([N_total], device=device, dtype=torch.long)
    neighbors_idx = torch.full((N_total, max_neighbors), shadow_index.item(), dtype=torch.long, device=device)

    for i in range(N_total):
        cand = torch.nonzero(within[i], as_tuple=False).flatten()
        # 優先取距離近的
        cand = cand[torch.argsort(dists[i, cand])]
        if cand.numel() > max_neighbors:
            cand = cand[:max_neighbors]
        if cand.numel() > 0:
            neighbors_idx[i, :cand.numel()] = cand

    return neighbors_idx


def collate_pair_like_d3feat(pts0: np.ndarray,
                             pts1: np.ndarray,
                             feat0: np.ndarray = None,
                             feat1: np.ndarray = None,
                             radius: float = 0.1,
                             max_neighbors: int = 32):
    """
    將兩個點雲打包為接近 D3Feat/KPConv 的 batch 字典，至少提供：
    - features: [N0+N1, C_in]
    - neighbors: list，僅第 0 層，形狀 [N0+N1, K]
    - stack_lengths: list，僅第 0 層，tensor([N0, N1])
    - points: list，僅第 0 層，形狀 [N0+N1, 3]

    參數：
    - pts0, pts1: numpy，皆為 [N,3]
    - feat0, feat1: numpy，若為 None 則自動用全 1 的 [N,1]
    - radius: 半徑鄰域
    - max_neighbors: 每點最多鄰居數 K
    """
    assert pts0.ndim == 2 and pts0.shape[1] == 3
    assert pts1.ndim == 2 and pts1.shape[1] == 3

    N0, N1 = pts0.shape[0], pts1.shape[0]

    if feat0 is None:
        feat0 = np.ones((N0, 1), dtype=np.float32)
    if feat1 is None:
        feat1 = np.ones((N1, 1), dtype=np.float32)

    # 組裝張量
    batched_points = torch.from_numpy(np.concatenate([pts0, pts1], axis=0)).float()
    batched_features = torch.from_numpy(np.concatenate([feat0, feat1], axis=0)).float()
    batched_lengths = torch.tensor([N0, N1], dtype=torch.long)

    # 建立第 0 層 neighbors
    neighbors_l0 = _radius_neighbors(batched_points, batched_lengths, radius, max_neighbors)

    # 與 D3Feat 相容的輸出結構
    dict_inputs = {
        'points': [batched_points],                 # List of [N_total, 3]
        'neighbors': [neighbors_l0],               # List of [N_total, K]
        'pools': [torch.zeros((0, 1), dtype=torch.long)],
        'upsamples': [torch.zeros((0, 1), dtype=torch.long)],
        'features': batched_features,              # [N_total, C_in]
        'stack_lengths': [batched_lengths],        # List of tensor([N0, N1])
    }
    return dict_inputs 