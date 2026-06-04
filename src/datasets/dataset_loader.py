import torch
from functools import partial

from datasets.dataset import GSDataset, JointDataset, build_joint_items
from util.util import seed_worker_fn, build_generator

def collate(all_data, matcher_type):
    # Ignore samples with no pts
    data = []
    for d in all_data:
        if d is None:
            continue
        if d.get("name", None) is None:
            continue
        if "pts2d" in d and len(d["pts2d"]) > 0:
            data.append(d)

    # Batch data contents
    batched = dict(
        name=[d["name"] for d in data],
        # 2D
        pts2d=torch.cat([torch.from_numpy(d["pts2d"]) for d in data]), # normalized 2d -> model 
        pts2d_pix=torch.cat([torch.from_numpy(d["pts2d_pix"]) for d in data]), # original 2d
        pts2dm=torch.cat([torch.from_numpy(d["pts2dm"]) for d in data]), # 2d go to model (normalized / bv)
        # desc2d=torch.cat([torch.from_numpy(d["desc2d"]) for d in data]),
        # 3D
        # gs3d=torch.cat([torch.from_numpy(d["gs3d"]) for d in data]),# normalized gs -> model
        pts3d=torch.cat([torch.from_numpy(d["pts3d"]) for d in data]),# original 3d
        pts3dm=torch.cat([torch.from_numpy(d["pts3dm"]) for d in data]), # 3d go to model (normalized / bv)
        idx2d=torch.cat(
            [
                torch.full((len(d["pts2dm"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        ),
        idx3d=torch.cat(
            [
                torch.full((len(d["pts3dm"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        ),
        matches_bin=torch.cat(
            [torch.from_numpy(d["matches_bin"]).view(-1) for d in data]
        ),
        R=torch.stack([torch.from_numpy(d["R"]) for d in data]),
        t=torch.stack([torch.from_numpy(d["t"]) for d in data]),
        K=torch.stack([torch.from_numpy(d["K"]) for d in data]),
        width=torch.tensor([d["width"] for d in data], dtype=torch.long),
        height=torch.tensor([d["height"] for d in data], dtype=torch.long),
    )
    if matcher_type != "GoMatch":
        batched["desc2d"] = torch.cat([
            torch.from_numpy(d["desc2d"]) for d in data
        ])
    return batched


def init_data_loader(opt, split):
    is_training = 'train' in split

    raw_dataset = GSDataset(opt, gs_root=opt.gs_root_path, split=split, return_params=True)

    items = build_joint_items(raw_dataset, match_root=opt.match_root, xfeat_root=opt.xfeat_root)
    dataset = JointDataset(opt, raw_dataset, items, h5_cache_size=16, drop_on_transient_error=True)

    print(f'The number of {split} data is: {len(dataset)}')

    g = build_generator(opt.seed)
    worker_init_fn = seed_worker_fn(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=is_training,
        num_workers=opt.num_workers,
        drop_last=is_training,
        worker_init_fn=worker_init_fn,
        collate_fn=partial(collate, matcher_type=opt.matcher_type),
        generator=g,
        persistent_workers=True if opt.num_workers > 0 else False,
        prefetch_factor=1 if opt.num_workers > 0 else None,
        pin_memory=True,
    )
