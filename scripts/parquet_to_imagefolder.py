"""Stanford Cars parquet -> ImageFolder 目录结构。

parquet 来自 HF 数据集 tanganke/stanford_cars，字段：
    image: {bytes, path}   label: int (ClassLabel)

输出（文件夹名 = "品牌__车型"，与 inference/pipeline.py 的解析约定一致）：
    data/stanford_cars/train/<品牌|车型>/*.jpg   90%
    data/stanford_cars/val/<品牌|车型>/*.jpg     10%
    data/stanford_cars/test/<品牌|车型>/*.jpg    （来自 test 分片）

用法：
    python scripts/parquet_to_imagefolder.py --src data/raw/stanford --out data/stanford_cars
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def read_class_names(path: Path) -> list[str]:
    """从 parquet 的 arrow schema 元数据里读出 ClassLabel 的类别名。"""
    schema = pq.read_schema(path)
    meta = schema.metadata or {}
    raw = meta.get(b"huggingface")
    if raw:
        info = json.loads(raw.decode("utf-8"))
        feats = info.get("info", {}).get("features", {})
        for key in ("label", "labels", "class"):
            if key in feats and "names" in feats[key]:
                return list(feats[key]["names"])
    return []


MULTI_WORD_BRANDS = [
    "Mercedes-Benz", "Aston Martin", "Alfa Romeo", "Land Rover", "Rolls-Royce",
    "AM General", "Great Wall", "Austin Healey", "MG", "Smart",
]


def split_label(name: str) -> str:
    """"Acura Integra Type R 2001" -> "Acura__Integra Type R"（去年份，品牌取首词）。"""
    parts = name.split()
    if not parts:
        return "unknown|unknown"
    brand, rest = parts[0], parts[1:]
    # 多词品牌优先匹配（如 AM General / Aston Martin）
    for mw in sorted(MULTI_WORD_BRANDS, key=lambda s: -len(s.split())):
        words = mw.split()
        if parts[: len(words)] == words:
            brand, rest = mw, parts[len(words):]
            break
    model_parts = [p for p in rest if not p.isdigit()]
    model = " ".join(model_parts) if model_parts else "unknown"
    # 文件名不能含 / \ : * ? " < > |
    for ch in '/\\:*?"<>':
        brand, model = brand.replace(ch, ""), model.replace(ch, "")
    return f"{brand}__{model}"


def dump(path: Path, out_root: Path, split: str, names: list[str], val_ratio: float = 0.1, seed: int = 42) -> int:
    table = pq.read_table(path, columns=["image", "label"])
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()

    random.seed(seed)
    count = 0
    for i, (img, lab) in enumerate(zip(images, labels)):
        cls = names[lab] if lab < len(names) else f"class_{lab}"
        target_split = split
        if split == "train" and random.random() < val_ratio:
            target_split = "val"
        cls_dir = out_root / target_split / split_label(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)
        # parquet 的 path 字段不唯一，统一按「分片名_行号」命名，避免覆盖
        name = f"{path.stem}_{i:06d}.jpg"
        (cls_dir / name).write_bytes(img["bytes"])
        count += 1
    print(f"  {path.name} -> {split}: {count} 张")
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/stanford"))
    ap.add_argument("--out", type=Path, default=Path("data/stanford_cars"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    train_files = sorted(args.src.glob("train-*.parquet"))
    test_files = sorted(args.src.glob("test-*.parquet"))
    if not train_files:
        raise SystemExit(f"{args.src} 下没有 train-*.parquet")

    names = read_class_names(train_files[0])
    print(f"类别数: {len(names)}")

    total = 0
    for f in train_files:
        total += dump(f, args.out, "train", names, args.val_ratio)
    for f in test_files:
        total += dump(f, args.out, "test", names, 0.0)

    print(f"[done] 共 {total} 张，输出目录 {args.out}")
    for split in ("train", "val", "test"):
        d = args.out / split
        if d.exists():
            n_cls = len([p for p in d.iterdir() if p.is_dir()])
            n_img = sum(1 for _ in d.rglob("*.jpg"))
            print(f"  {split}: {n_cls} 类 / {n_img} 张")


if __name__ == "__main__":
    main()
