"""真实场景跨域评估：用网络实拍照片检验品牌分类器（而非 Stanford Cars 的验证集）。

Stanford Cars 的官方划分存在「同一辆车多张照片」跨集合泄漏，验证集分数会偏高。
本脚本用 Wikimedia 上不同来源、不同光线的实拍图，给出更接近真实部署的指标。

用法：
    python scripts/eval_realworld.py
    python scripts/eval_realworld.py --dir test_images/realworld
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def label_brand(label: str) -> str:
    """从 'BMW__5 Series Sedan' 这类标签里取品牌。"""
    sep = "__" if "__" in label else "|"
    return label.split(sep)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "test_images" / "realworld")
    ap.add_argument("--show-top", type=int, default=3)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(ROOT))
    from inference.pipeline import VehicleRecognizer

    labels_file = args.dir / "labels.json"
    meta = json.loads(labels_file.read_text(encoding="utf-8")) if labels_file.exists() else {}

    rec = VehicleRecognizer()
    if rec.classifier is None:
        raise SystemExit("分类器未加载，请先训练 weights/vehicle_classifier.pth")

    rows, brand_ok, model_ok = [], 0, 0
    for gt_brand, info in meta.items():
        path = args.dir / info["file"]
        if not path.exists():
            continue
        result = rec.recognize(str(path)).to_dict()
        v = result["vehicle"]
        pred_brand, pred_model = v["brand"], v["model"]
        rows.append(
            {
                "file": info["file"],
                "gt_brand": gt_brand,
                "pred_brand": pred_brand,
                "pred_model": pred_model,
                "conf": round(v["confidence"], 4),
                "plate": result["plate"]["text"],
                "brand_hit": pred_brand == gt_brand,
            }
        )
        brand_ok += pred_brand == gt_brand

    if not rows:
        raise SystemExit("没有可评估的图片")

    n = len(rows)
    print(f"\n{'文件':28s}{'真值品牌':16s}{'预测品牌':16s}{'预测车型':26s}置信度")
    print("-" * 96)
    for r in rows:
        flag = "  " if r["brand_hit"] else "✗ "
        print(f"{flag}{r['file'][:26]:28s}{r['gt_brand']:16s}{r['pred_brand']:16s}"
              f"{r['pred_model'][:24]:26s}{r['conf']:.3f}")

    acc = brand_ok / n
    print("-" * 96)
    print(f"真实照片品牌 Top-1 准确率：{brand_ok}/{n} = {acc:.2%}")

    out = ROOT / "docs" / "realworld_eval.md"
    out.parent.mkdir(exist_ok=True)
    lines = [
        "# 真实场景跨域评估（品牌分类）",
        "",
        f"- 数据来源：Wikimedia Commons 实拍照片，共 {n} 张、{len({r['gt_brand'] for r in rows})} 个品牌",
        f"- 品牌 Top-1 准确率：**{acc:.2%}**（{brand_ok}/{n}）",
        "- 说明：Stanford Cars 官方划分存在同车多图跨集合泄漏，验证集分数偏高；本表为独立来源实拍数据，更接近真实部署表现。",
        "",
        "| 文件 | 真值品牌 | 预测品牌 | 预测车型 | 置信度 | 是否正确 |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['file']} | {r['gt_brand']} | {r['pred_brand']} | {r['pred_model']} | "
        f"{r['conf']:.3f} | {'✅' if r['brand_hit'] else '❌'} |"
        for r in rows
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入：{out}")


if __name__ == "__main__":
    main()
