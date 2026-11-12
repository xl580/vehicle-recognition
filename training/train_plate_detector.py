"""YOLOv8 车牌检测微调（CCPD 数据集）。

前置：先运行 scripts/prepare_ccpd.py 生成 YOLO 格式数据集：

    data/plate_det/
    ├── images/{train,val}
    ├── labels/{train,val}
    └── data.yaml

用法：
    python training/train_plate_detector.py --data data/plate_det/data.yaml --epochs 50

产出：
    runs/detect/plate/weights/best.pt  ->  复制到 weights/plate_detector.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "plate_det" / "data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="预训练权重，可选 yolov8n/s/m")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="plate")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"找不到数据集配置：{args.data}\n请先运行 scripts/prepare_ccpd.py")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.bs,
        lr0=args.lr0,
        name=args.name,
        project=str(ROOT / "runs" / "detect"),
        patience=15,
        cos_lr=True,
        close_mosaic=10,
        workers=args.workers,
        amp=True,
        plots=True,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        WEIGHTS.mkdir(exist_ok=True)
        shutil.copy2(best, WEIGHTS / "plate_detector.pt")
        print(f"[done] 最佳权重已复制到 {WEIGHTS / 'plate_detector.pt'}")

    metrics = getattr(results, "results_dict", {}) or {}
    for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
        if k in metrics:
            print(f"  {k} = {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
