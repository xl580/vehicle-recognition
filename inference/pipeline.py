"""VehicleRecognizer：车牌 / 品牌 / 车型识别核心流水线。

流程：
    1. YOLOv8 检测车辆（COCO: car/truck/bus）
    2. 裁剪车辆 -> YOLOv8 车牌检测 -> HyperLPR3 字符识别
    3. 裁剪车辆 -> ResNet50 品牌/车型分类
    4. 结果融合输出 JSON

用法：
    from inference.pipeline import VehicleRecognizer
    rec = VehicleRecognizer()          # 自动加载权重
    result = rec.recognize("test.jpg")
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---- 可选依赖：缺失时对应模块自动降级，保证流水线可部分运行 ----
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from hyperlpr3 import LicensePlateCatcher
except ImportError:
    LicensePlateCatcher = None

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:
    torch = None

try:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = CLIPProcessor = None

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

# COCO 中属于"车辆"的类别 id
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# 中文车牌类型（严格对应 hyperlpr3/common/typedef.py 的枚举顺序）
PLATE_TYPES = [
    "蓝牌",         # 0
    "黄牌单层",     # 1
    "白牌单层",     # 2
    "绿牌新能源",   # 3
    "黑牌港澳",     # 4
    "香港单层",     # 5
    "香港双层",     # 6
    "澳门单层",     # 7
    "澳门双层",     # 8
    "黄牌双层",     # 9
]

# 品牌/车型标签（训练时由训练脚本生成 label_map.json）
LABEL_MAP_PATH = WEIGHTS / "label_map.json"


@dataclass
class RecognizeResult:
    """单张图片的识别结果。"""

    plate_text: str = ""
    plate_type: str = ""
    plate_conf: float = 0.0
    plate_box: list[int] = field(default_factory=list)
    brand: str = ""
    model: str = ""
    vehicle_conf: float = 0.0
    vehicle_box: list[int] = field(default_factory=list)
    color: str = ""
    time_cost_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate": {
                "text": self.plate_text,
                "plate_type": self.plate_type,
                "confidence": round(self.plate_conf, 4),
                "box": self.plate_box,
            },
            "vehicle": {
                "brand": self.brand,
                "model": self.model,
                "color": self.color,
                "confidence": round(self.vehicle_conf, 4),
                "box": self.vehicle_box,
            },
            "time_cost_ms": {k: round(v, 1) for k, v in self.time_cost_ms.items()},
        }


class VehicleClassifier:
    """ResNet50 品牌/车型分类器（training/train_classifier.py 训练产出）。"""

    def __init__(self, weights_path: Path, num_classes: int, label_map: dict[str, int]):
        if torch is None:
            raise RuntimeError("未安装 torch/torchvision，无法加载分类器")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state if isinstance(state, dict) and "model" not in state else state["model"])
        self.model.to(self.device).eval()
        self.idx2label = {v: k for k, v in label_map.items()}
        # 与 training/train_classifier.py 的 val 预处理保持完全一致：
        # Resize(256) + CenterCrop(224)，且必须是 RGB（训练用的是 PIL 读图）
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict(self, bgr: np.ndarray, allow: set[str] | None = None) -> tuple[str, float]:
        # OpenCV 是 BGR，ToPILImage 按 RGB 解释，必须先换通道，否则红蓝互换
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0]
        if allow is not None:
            # 品牌约束解码：只在给定品牌的类别里取最大（品牌由 CLIP 提供）
            mask = torch.tensor(
                [self.idx2label.get(i, "") in allow for i in range(probs.shape[0])],
                device=probs.device,
            )
            if mask.any():
                probs = probs * mask
        conf, idx = probs.max(0)
        return self.idx2label.get(int(idx), "unknown"), float(conf)

    def brands(self) -> set[str]:
        """所有类别所属的品牌集合。"""
        return {label_brand(l) for l in self.idx2label.values()}

    def classes_of(self, brand: str) -> set[str]:
        """某品牌下的所有（车型）类别名。"""
        return {l for l in self.idx2label.values() if label_brand(l) == brand}


def label_brand(label: str) -> str:
    """从 'BMW__5 Series Sedan' / 'BMW|5 Series Sedan' 取品牌；品牌级标签原样返回。"""
    if "__" in label or "|" in label:
        sep = "__" if "__" in label else "|"
        return label.split(sep)[0]
    return label


class ClipBrandClassifier:
    """CLIP 视觉特征 + 线性探针的品牌分类器。

    ResNet50 在 Stanford Cars（棚拍图）上微调，换街拍会失效（背景捷径学习）；
    CLIP 特征跨域鲁棒得多，用它做品牌判别，车型交给 ResNet50 细粒度模型。
    """

    def __init__(self, clip_dir: Path, probe_path: Path):
        if CLIPModel is None or torch is None:
            raise RuntimeError("未安装 transformers/torch，无法加载 CLIP 品牌分类器")
        probe = np.load(probe_path, allow_pickle=True)
        self.W: np.ndarray = probe["W"].astype(np.float32)
        self.b: np.ndarray = probe["b"].astype(np.float32)
        self.classes: list[str] = [str(c) for c in probe["classes"]]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(str(clip_dir)).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(str(clip_dir))

    @torch.no_grad()
    def predict(self, bgr_list: list[np.ndarray]) -> tuple[str, float]:
        """对多张视图（整图 + 车辆裁剪）取特征平均，兼得全局上下文与主体聚焦。"""
        from PIL import Image

        imgs = [Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in bgr_list if b is not None and b.size]
        inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.mean(dim=0)
        feat = (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
        logits = self.W @ feat + self.b
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])


def estimate_color(bgr: np.ndarray) -> str:
    """HSV 统计估计车身主色（轻量方案）。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s > 60) & (v > 60)
    if mask.mean() < 0.05 or (s[mask] < 90).mean() > 0.6:
        return "白色" if v.mean() > 120 else "黑色/深灰"
    hue = np.median(h[mask])
    if hue < 10 or hue >= 160:
        return "红色"
    if hue < 25:
        return "橙色/黄色"
    if hue < 40:
        return "黄色"
    if hue < 90:
        return "绿色"
    if hue < 130:
        return "蓝色"
    return "紫色/其他"


class VehicleRecognizer:
    """端到端识别流水线。各模块按可用性自动降级。"""

    def __init__(
        self,
        vehicle_model: str = "yolov8s.pt",
        plate_model: str | None = None,
        classifier_weights: str | None = None,
        device: str | int | None = None,
    ):
        self.det_vehicle = None
        self.det_plate = None
        self.ocr = None
        self.classifier = None
        # 显式指定推理设备：有 CUDA 就用 GPU（0），否则 CPU
        if device is not None:
            self.device: str | int = device
        else:
            self.device = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"

        if YOLO is not None:
            # 优先使用本地 weights/ 下的权重，避免离线环境下触发 GitHub 下载
            vm = Path(vehicle_model)
            if not vm.exists():
                local = WEIGHTS / vm.name
                vm = local if local.exists() else vm
            self.det_vehicle = YOLO(str(vm))

            plate_path = Path(plate_model) if plate_model else WEIGHTS / "plate_detector.pt"
            if plate_path.exists():
                self.det_plate = YOLO(str(plate_path))

        if LicensePlateCatcher is not None:
            # HyperLPR3 自带检测+识别，作为车牌链路主力或 fallback
            self.ocr = LicensePlateCatcher()

        cls_path = Path(classifier_weights) if classifier_weights else WEIGHTS / "vehicle_classifier.pth"
        if cls_path.exists() and LABEL_MAP_PATH.exists() and torch is not None:
            import json

            label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
            self.classifier = VehicleClassifier(cls_path, num_classes=len(label_map), label_map=label_map)

        # CLIP 品牌分类器（可选）：与细粒度车型模型组成双模型结构
        self.brand_clf = None
        probe_path = WEIGHTS / "clip_brand_probe.npz"
        clip_dir = ROOT / "models" / "clip-vit-base-patch32"
        if probe_path.exists() and clip_dir.exists() and CLIPModel is not None:
            try:
                self.brand_clf = ClipBrandClassifier(clip_dir, probe_path)
            except Exception as e:  # 权重缺失/版本不兼容时降级
                print(f"[warn] CLIP 品牌分类器加载失败，降级为 ResNet50 品牌输出: {e}")
                self.brand_clf = None

    # ------------------------------------------------------------------
    def recognize(self, image: str | np.ndarray) -> RecognizeResult:
        t0 = time.perf_counter()
        img = cv2.imread(image) if isinstance(image, str) else image
        if img is None:
            raise ValueError("无法读取图像")

        res = RecognizeResult()

        vehicle_box, vehicle_crop = self._detect_vehicle(img)
        res.vehicle_box = vehicle_box
        res.time_cost_ms["detection"] = (time.perf_counter() - t0) * 1000

        if vehicle_crop is not None:
            if self.classifier is not None or self.brand_clf is not None:
                t = time.perf_counter()
                brand, model, conf = self._classify(vehicle_crop, img)
                res.brand, res.model, res.vehicle_conf = brand, model, conf
                res.color = estimate_color(vehicle_crop)
                res.time_cost_ms["classify"] = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            self._recognize_plate(img, vehicle_box, res)
            res.time_cost_ms["ocr"] = (time.perf_counter() - t) * 1000

        return res

    # ------------------------------------------------------------------
    def _classify(self, crop: np.ndarray, full_img: np.ndarray | None = None) -> tuple[str, str, float]:
        """双模型分类：品牌用 CLIP 探针（跨域稳），车型用 ResNet50（细粒度）+ 品牌约束解码。"""
        brand, b_conf = (None, 0.0)
        if self.brand_clf is not None:
            brand, b_conf = self.brand_clf.predict([crop, full_img])

        if self.classifier is None:
            return (brand or "unknown"), "unknown", b_conf

        allow = self.classifier.classes_of(brand) if brand else None
        label, m_conf = self.classifier.predict(crop, allow=allow)
        l_brand = label_brand(label)
        if allow is not None and l_brand != brand:
            # CLIP 判的品牌在车型模型里没有对应类别，退回无约束解码
            label, m_conf = self.classifier.predict(crop)
            l_brand = label_brand(label)
        brand = brand or l_brand
        model = self._model_of(label) or label
        return brand, model, (b_conf if self.brand_clf is not None else m_conf)

    @staticmethod
    def _model_of(label: str) -> str:
        sep = "__" if "__" in label else "|"
        _, _, model = label.partition(sep)
        return model

    # ------------------------------------------------------------------
    def _detect_vehicle(self, img: np.ndarray) -> tuple[list[int], np.ndarray | None]:
        if self.det_vehicle is None:
            return [], img  # 降级：整图作为车辆区域
        out = self.det_vehicle.predict(img, verbose=False, conf=0.4, device=self.device)[0]
        best, best_conf = None, 0.0
        for box in out.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_IDS and float(box.conf[0]) > best_conf:
                best, best_conf = box, float(box.conf[0])
        if best is None:
            return [], img
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return [x1, y1, x2, y2], img[y1:y2, x1:x2].copy()

    def _recognize_plate(self, img: np.ndarray, vehicle_box: list[int], res: RecognizeResult) -> None:
        # 路线 A：YOLO 车牌检测 + HyperLPR 识别
        if self.det_plate is not None and self.ocr is not None:
            out = self.det_plate.predict(img, verbose=False, conf=0.35, device=self.device)[0]
            if len(out.boxes):
                box = out.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 检测框常紧贴字符，向外扩 8%，避免边缘字符被裁掉
                pad_x = int((x2 - x1) * 0.08)
                pad_y = int((y2 - y1) * 0.08)
                h, w = img.shape[:2]
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                crop = img[y1:y2, x1:x2]
                ocr_res = self.ocr(crop) if crop.size else []
                if ocr_res and float(ocr_res[0][1]) >= 0.9:  # 裁剪图识别可信才采用
                    text, conf, ptype, *_ = ocr_res[0]
                    res.plate_text, res.plate_conf = text, float(conf)
                    res.plate_type = PLATE_TYPES[int(ptype)] if int(ptype) < len(PLATE_TYPES) else ""
                    res.plate_box = [x1, y1, x2, y2]
                    return
        # 路线 B：仅 HyperLPR（自带检测）
        if self.ocr is not None:
            results = self.ocr(img)
            if results:
                text, conf, ptype, box, *_ = results[0]
                res.plate_text, res.plate_conf = text, float(conf)
                res.plate_type = PLATE_TYPES[int(ptype)] if int(ptype) < len(PLATE_TYPES) else ""
                res.plate_box = _poly_to_xyxy(box)


def _poly_to_xyxy(box) -> list[int]:
    """HyperLPR3 返回 4 角点（8 值）或 xyxy（4 值），统一转为 xyxy 矩形。"""
    nums = [float(v) for v in box]
    if len(nums) >= 8:
        xs, ys = nums[0::2], nums[1::2]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    return [int(v) for v in nums[:4]]


if __name__ == "__main__":
    import sys

    rec = VehicleRecognizer()
    result = rec.recognize(sys.argv[1] if len(sys.argv) > 1 else "test.jpg")
    import json

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
