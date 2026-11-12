"""模型导出为 ONNX，用于生产环境加速推理。

用法：
    python scripts/export_onnx.py --weights weights/vehicle_classifier.pth --num-classes 431

产出：
    weights/vehicle_classifier.onnx

YOLOv8 权重导出更简单：
    from ultralytics import YOLO
    YOLO("weights/plate_detector.pt").export(format="onnx", opset=12, simplify=True, dynamic=True)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "vehicle_classifier.pth")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "weights" / "vehicle_classifier.onnx")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"找不到权重：{args.weights}")

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"[done] 已导出 {args.out} ({size_mb:.1f} MB)")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        import numpy as np

        out = sess.run(None, {"input": dummy.numpy()})[0]
        with torch.no_grad():
            ref = model(dummy).numpy()
        diff = float(np.abs(out - ref).max())
        print(f"[verify] ONNX 与 PyTorch 最大误差 {diff:.2e}（< 1e-3 视为通过）")
    except ImportError:
        print("[skip] 未安装 onnxruntime，跳过一致性校验")


if __name__ == "__main__":
    main()
