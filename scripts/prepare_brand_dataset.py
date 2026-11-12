"""构建「品牌级」分类数据集，合并 Stanford Cars（棚拍）与 TAU Vehicle（网络实拍）。

做品牌分类而不做车型分类的原因：Stanford Cars 的图几乎都是展厅/拍卖棚拍，背景单一，
模型容易学到背景线索；一旦换成街拍就崩（实测从 90% 掉到 8%）。把两个域的数据合起来
按品牌标签训练，才能让品牌输出在真实场景可用。

用法：
    python scripts/prepare_brand_dataset.py \
        --stanford data/stanford_cars --tau data/raw/tau_extract --out data/brand_cls

产出：
    data/brand_cls/{train,val}/<品牌>/xxxx.jpg
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

# 多词品牌，必须优先匹配，否则 "Land Rover" 会被拆成 "Land"
MULTIWORD = [
    "Mercedes-Benz", "Land Rover", "Aston Martin", "Alfa Romeo", "Rolls-Royce",
    "Great Wall", "Bentley", "Bugatti", "Lamborghini", "Maserati", "Maybach",
    "Ssangyong", "Citroen", "Infiniti", "Acura", "Genesis", "Dodge", "RAM",
]

# 不同数据集里的写法归一
ALIAS = {
    "VW": "Volkswagen",
    "Chevy": "Chevrolet",
    "Benz": "Mercedes-Benz",
    "Mercedes": "Mercedes-Benz",
}


def norm_brand(raw: str) -> str:
    """把各种命名风格归一成品牌名。"""
    s = raw.replace("_", " ").replace("-", "-").strip()
    for b in MULTIWORD:
        if s.lower().startswith(b.lower()):
            return ALIAS.get(b, b)
    token = s.split()[0] if s.split() else s
    return ALIAS.get(token, token)


def brand_from_stanford(folder_name: str) -> str:
    """'BMW__5 Series Sedan' -> 'BMW'"""
    sep = "__" if "__" in folder_name else "|"
    return norm_brand(folder_name.split(sep)[0])


def brand_from_tau(folder_name: str) -> str:
    """'Audi_A5_Coupe_2011' / 'Audi A5 Coupe 2011' -> 'Audi'"""
    return norm_brand(folder_name)


def collect(root: Path, to_brand, max_per_brand: int, rng: random.Random) -> dict[str, list[Path]]:
    """扫描 root 下「子目录=类别」的结构，返回 {品牌: [图片路径]}（已按上限采样）。"""
    pool: dict[str, list[Path]] = defaultdict(list)
    for cls_dir in sorted([p for p in root.rglob("*") if p.is_dir()]):
        imgs = [p for p in cls_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not imgs:
            continue
        brand = to_brand(cls_dir.name)
        if not brand or brand.lower() in ("unknown", "others"):
            continue
        pool[brand].extend(imgs)

    out = {}
    for brand, files in pool.items():
        rng.shuffle(files)
        out[brand] = files[:max_per_brand]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stanford", type=Path, default=Path("data/stanford_cars"))
    ap.add_argument("--tau", type=Path, default=Path("data/raw/tau_extract"))
    ap.add_argument("--out", type=Path, default=Path("data/brand_cls"))
    ap.add_argument("--max-per-brand", type=int, default=400)
    ap.add_argument("--min-per-brand", type=int, default=40)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    merged: dict[str, list[Path]] = defaultdict(list)

    if args.stanford.exists():
        for split in ("train", "val"):
            d = args.stanford / split
            if d.exists():
                for brand, files in collect(d, brand_from_stanford, args.max_per_brand // 2, rng).items():
                    merged[brand].extend(files)
        print(f"[stanford] 品牌数 {len(merged)}")

    if args.tau.exists():
        tau_pool = collect(args.tau, brand_from_tau, args.max_per_brand, rng)
        new_brands = 0
        for brand, files in tau_pool.items():
            if brand not in merged:
                new_brands += 1
            merged[brand].extend(files)
        print(f"[tau] 品牌数 {len(tau_pool)}，其中新增 {new_brands}")
    else:
        print(f"[warn] 未找到 TAU 目录 {args.tau}，仅用 Stanford 数据")

    # 过滤样本过少的品牌，避免无法划分验证集
    kept = {b: f for b, f in merged.items() if len(f) >= args.min_per_brand}
    dropped = len(merged) - len(kept)
    print(f"[info] 保留品牌 {len(kept)} 个，丢弃样本过少品牌 {dropped} 个")

    out = args.out
    if out.exists():
        shutil.rmtree(out)

    total_tr = total_va = 0
    for brand, files in sorted(kept.items()):
        safe = re.sub(r'[\\/:*?"<>|]', "", brand)
        rng.shuffle(files)
        n_val = max(2, int(len(files) * args.val_ratio))
        for split, part in (("val", files[:n_val]), ("train", files[n_val:])):
            d = out / split / safe
            d.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(part):
                shutil.copy2(src, d / f"{src.stem[:40]}_{i:04d}{src.suffix.lower()}")
            total_tr += len(part) if split == "train" else 0
            total_va += len(part) if split == "val" else 0

    print(f"[done] 品牌 {len(kept)} 个 | train {total_tr} 张 / val {total_va} 张 -> {out}")


if __name__ == "__main__":
    main()
