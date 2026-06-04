import argparse
import random
from pathlib import Path
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def load_pool(root: Path, seed: int):
    """Return remaining pool: {cat: [Path,...]} shuffled deterministically."""
    rng = random.Random(seed)
    pool = {
        d.name: list(d.glob("*.ply"))
        for d in root.iterdir()
        if d.is_dir() and d.name.isdigit()
    }
    # drop empty folders
    pool = {k: v for k, v in pool.items() if v}
    # shuffle each category list (reproducible)
    for k in pool:
        rng.shuffle(pool[k])
    return pool


def take_balanced(pool, n: int, rng: random.Random):
    """
    Take n samples balanced across categories, WITHOUT replacement.
    Mutates pool (consumes picked items).
    Returns (picked_paths, counts_dict).
    """
    available = sum(len(v) for v in pool.values())
    if n > available:
        n = available  # okay to take all

    keys = [k for k, v in pool.items() if v]
    keys.sort(key=int)  # stable order
    rng.shuffle(keys)   # but still randomize for fairness

    counts = {k: 0 for k in keys}
    picked = []

    # base share
    if keys:
        base = n // len(keys)
    else:
        return [], {}

    rem = n
    for k in keys:
        if rem <= 0:
            break
        take = min(base, len(pool[k]))
        if take:
            picked.extend(pool[k][:take])
            pool[k] = pool[k][take:]
            counts[k] += take
            rem -= take

    # round-robin remainder
    while rem > 0:
        progressed = False
        for k in keys:
            if rem <= 0:
                break
            if pool[k]:
                # pop from end (still random because list was shuffled)
                picked.append(pool[k].pop())
                counts[k] += 1
                rem -= 1
                progressed = True
        if not progressed:
            break

    rng.shuffle(picked)
    return picked, counts

def read_selected_files(txt_path: Path):
    # each line is a filename like: 02843684-xxxx.ply
    selected = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                selected.add(name)
    return selected


def folder_to_count(root: Path):
    """{cat: total_file_count} only numeric folders"""
    counts = {}
    for d in root.iterdir():
        if d.is_dir() and d.name.isdigit():
            counts[d.name] = len(list(d.glob("*.ply")))
    return counts


def plot_per_category_stacked(
    root_dir: str,
    split_txts: dict,
    out_png: str = None,
    use_ratio: bool = True,
    split_rename: dict = None,
    # total display options:
    total_mode: str = "line",     # "line" | "topk" | "every" | "off"
    topk: int = 10,               # used when total_mode="topk"
    every: int = 5,               # used when total_mode="every"
    # color options:
    colors: dict = None,          # {"train":"#...", "val":"#...", "test":"#...", "remaining":"#..."}
):
    root = Path(root_dir)
    split_rename = split_rename or {}

    # ---- pleasant, readable default colors (change if you like) ----
    if colors is None:
        colors = {
            "train": "#4C78A8",      # muted blue
            "val": "#F58518",        # orange
            "test": "#54A24B",       # green
            "remaining": "#BDBDBD",  # light grey
            "total_line": "#333333", # dark grey
        }

    totals = folder_to_count(root)
    cats = sorted(totals.keys(), key=int)

    # counts per split per category
    split_counts = {s: {c: 0 for c in cats} for s in split_txts.keys()}
    for split_name, txt in split_txts.items():
        for fn in read_selected_files(Path(txt)):
            cat = fn.split("-", 1)[0]
            if cat in totals:
                split_counts[split_name][cat] += 1

    # build stacked heights
    denom = np.array([max(1, totals[c]) for c in cats], dtype=float)
    stack_vals = {}
    for s in split_txts.keys():
        arr = np.array([split_counts[s][c] for c in cats], dtype=float)
        stack_vals[s] = (arr / denom) if use_ratio else arr

    used = np.zeros(len(cats), dtype=float)
    for s in split_txts.keys():
        used += stack_vals[s]

    remaining = np.maximum(0.0, 1.0 - used) if use_ratio else None

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(14, len(cats) * 0.25), 6))

    bottom = np.zeros(len(cats), dtype=float)

    # plot splits in order given by split_txts
    for s in split_txts.keys():
        label = split_rename.get(s, s)
        # map label to a color if provided, otherwise fallback
        c = colors.get(label, None)
        ax.bar(x, stack_vals[s], bottom=bottom, width=0.9, label=label, color=c)
        bottom += stack_vals[s]

    # remaining (grey)
    if use_ratio:
        ax.bar(
            x,
            remaining,
            bottom=bottom,
            width=0.9,
            label="Remaining",
            color=colors["remaining"],
            alpha=0.8,
        )

    # axes
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=90, fontsize=8)
    ax.set_xlabel("Category folder")
    ax.set_ylabel("Ratio (selected / total)" if use_ratio else "Count")
    ax.set_title("Per-category stacked split composition")
    ax.legend(loc="upper right")

    # ---- show totals without overlap ----
    total_arr = np.array([totals[c] for c in cats], dtype=float)

    if total_mode == "line":
        ax2 = ax.twinx()
        ax2.plot(x, total_arr, linewidth=1.2, color=colors["total_line"], marker="o", markersize=2.5)
        ax2.set_ylabel("Total .ply per category")
        ax2.grid(False)

    elif total_mode == "topk":
        # label only top-k totals
        idx = np.argsort(-total_arr)[:topk]
        for i in idx:
            ax.text(i, 1.01 if use_ratio else bottom[i] + 0.01, f"{int(total_arr[i])}",
                    ha="center", va="bottom", fontsize=8, rotation=90)

    elif total_mode == "every":
        # label every N bars
        for i in range(0, len(cats), max(1, every)):
            ax.text(i, 1.01 if use_ratio else bottom[i] + 0.01, f"{int(total_arr[i])}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    # tidy
    fig.tight_layout()

    if out_png:
        out_path = Path(out_png).resolve()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")
        plt.close(fig)
    else:
        plt.show()
def plot_balance_counts(
    root_dir: str,
    split_txts: dict,
    out_png: str = None,
    split_rename: dict = None,
    colors: dict = None,
):
    root = Path(root_dir)
    split_rename = split_rename or {}

    # use shared color config
    if colors is None:
        colors = COLORS

    totals = folder_to_count(root)
    cats = sorted(totals.keys(), key=int)
    splits = list(split_txts.keys())

    # count selected files per split per category
    split_counts = {s: {c: 0 for c in cats} for s in splits}
    for s, txt in split_txts.items():
        for fn in read_selected_files(Path(txt)):
            cat = fn.split("-", 1)[0]
            if cat in totals:
                split_counts[s][cat] += 1

    # per-category total selected (all splits)
    per_cat_total = np.array([sum(split_counts[s][c] for s in splits) for c in cats], dtype=float)
    N = per_cat_total.sum()
    K = max(1, len(cats))
    ideal = N / K

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(14, len(cats) * 0.25), 5))

    # stacked counts by split (colored)
    bottom = np.zeros(len(cats), dtype=float)
    for s in splits:
        label = split_rename.get(s, s)          # e.g. "train"
        bar_color = colors.get(label, None)     # pulls COLORS["train"]/["val"]/["test"]
        heights = np.array([split_counts[s][c] for c in cats], dtype=float)
        ax.bar(x, heights, bottom=bottom, width=0.9, label=label, color=bar_color)
        bottom += heights

    # ideal line (same style/color config)
    ax.axhline(
        ideal,
        linestyle="--",
        linewidth=1.5,
        color=colors.get("ideal_line", "#191919"),
        label=f"Ideal ({ideal:.1f})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=90, fontsize=8)
    ax.set_xlabel("Category folder")
    ax.set_ylabel("Selected count")
    ax.set_title("Per-category sample counts with ideal reference")
    ax.legend(loc="upper right")

    fig.tight_layout()

    if out_png:
        out_path = Path(out_png).resolve()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved balance plot to: {out_path}")
        plt.close(fig)
    else:
        plt.show()
       
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=Path, default="/mnt/nct-zfs/TCO-All/SharedDatasets/ShapeSplat")
    ap.add_argument("--samples", type=int, nargs="+", required=True, help="e.g. --samples 50 1000 500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", default="selected", help="Outputs: <prefix>_<N>.txt")
    ap.add_argument("--plot_png", default=None, help="If set, save stacked plot to this png")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_pool(args.root_dir, args.seed)

    if not pool:
        raise RuntimeError("No numeric category folders with .ply files found")

    initial_totals = {k: len(v) for k, v in pool.items()}

    # record split txt paths for plotting
    split_txts = {}

    for n in args.samples:
        picked, counts = take_balanced(pool, n, rng)

        out_path = Path(f"{args.out_prefix}_{n}.txt")
        split_txts[f"{args.out_prefix}_{n}"] = str(out_path)  # for legend + reading

        with out_path.open("w", encoding="utf-8") as f:
            for p in picked:
                f.write(p.name + "\n")

        print(f"\n=== {out_path} ===")
        print(f"Seed: {args.seed} | Requested: {n} | Written: {len(picked)}")
        for k in sorted(initial_totals, key=int):
            c = counts.get(k, 0)
            print(f"{k}: {c}/{initial_totals[k]}")

    remaining = sum(len(v) for v in pool.values())
    print(f"\nRemaining unused after all files: {remaining}")

    # ---- Plot combined distribution across all generated splits ----
    plot_per_category_stacked(
        root_dir=str(args.root_dir),
        split_txts=split_txts,
        out_png=args.plot_png,
        use_ratio=True,
        split_rename={
            "selected_6743": "train",
            "selected_1124": "val",
            "selected_2623": "test",
        },
        total_mode="line",
        colors = {
            "train": "#1f6f6f",  
            "val": "#54a1a1",        
            "test": "#9fc8c8",       
            "remaining": "#E3E2E2",
            "total_line": "#191919", # dark grey
        }
        )
    plot_balance_counts(
    root_dir=str(args.root_dir),
    split_txts=split_txts,
    out_png=("balance_counts.png" if args.plot_png else None),
    split_rename={
        "selected_6743": "train",
        "selected_1124": "val",
        "selected_2623": "test",
    },
    colors = {
    "train": "#1f6f6f",  
    "val": "#54a1a1",        
    "test": "#9fc8c8",     
    "ideal_line": "#222222",
    "total_bar": "#A0A0A0",  

}
)

    


if __name__ == "__main__":
    main()
       

