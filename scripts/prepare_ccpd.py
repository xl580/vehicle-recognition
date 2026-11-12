"""把 CCPD 原始数据集转换为 YOLO 检测格式。

CCPD 文件名格式：
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
     │    │      │                │                                   │      │   └ 模糊度
     │    │      │                │                                   │      └ 亮度
     │    │      │                │                                   └ 车牌字符索引
     │    │      │                └ 车牌四个角点
     │    │      └ 车牌 bbox: x1&y1_x2&y2
     │    └ 水平/垂直倾斜角
     └ 车牌面积占比

用法：
    python scripts/prepare_ccpd.py --src /path/to/CCPD2019/ccpd_base --limit 20000
产出：
    data/plate_det/images/{train,val}/*.jpg
    data/plate_det/labels/{train,val}/*.txt
    data/plate_det/data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "plate_det"


def parse_bbox(filename: str) -> tuple[int, int, int, int] | None:
    """从 CCPD 文件名解析车牌 bbox（兼容 `&` 与 `,` 两种坐标分隔符）。"""
    try:
        box_part = filename.split("-")[2].replace(",", "&")
        p1, p2 = box_part.split("_")
        x1, y1 = map(int, p1.split("&"))
        x2, y2 = map(int, p2.split("&"))
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    except (IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    files = sorted([p for p in args.src.glob("*.jpg")])[: args.limit]
    if not files:
        raise SystemExit(f"{args.src} 下没有找到 jpg 文件")

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(files)
    n_val = int(len(files) * args.val_ratio)
    stats = {"ok": 0, "bad_name": 0, "bad_img": 0}

    for i, src in enumerate(files):
        split = "val" if i < n_val else "train"
        bbox = parse_bbox(src.name)
        if bbox is None:
            stats["bad_name"] += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            stats["bad_img"] += 1
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        # 归一化为 YOLO 的 cx, cy, w, h
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if not (0 < bw <= 1 and 0 < bh <= 1):
            stats["bad_name"] += 1
            continue

        shutil.copy2(src, OUT / "images" / split / src.name)
        (OUT / "labels" / split / f"{src.stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
        )
        stats["ok"] += 1

        if stats["ok"] % 2000 == 0:
            print(f"  已处理 {stats['ok']} 张...")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames: ['plate']\n",
        encoding="utf-8",
    )
    print(f"[done] 成功 {stats['ok']} 张，跳过（文件名异常 {stats['bad_name']} / 图片损坏 {stats['bad_img']}）")
    print(f"[done] 数据集配置：{OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
