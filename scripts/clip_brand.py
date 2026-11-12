"""CLIP 品牌识别：零样本 + 线性探针（跨域更稳）。

动机：ResNet50 在 Stanford Cars（棚拍图）上微调，验证集 90%，但换到街拍只有 8%~33%。
原因是它把背景当成了判别线索。CLIP 用 4 亿图文对预训练，视觉表征对背景/光照的变化
鲁棒得多，用它的特征做线性分类器，跨域表现通常远好于端到端微调的小数据集模型。

用法：
    python scripts/clip_brand.py --mode zero-shot          # 零样本评测
    python scripts/clip_brand.py --mode probe              # 训练线性探针并评测
    python scripts/clip_brand.py --mode probe --export weights/
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
# 权重已手动下载到本地目录，禁掉联网校验，避免 hf-mirror 的 API 限流（429）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

LOCAL_CLIP = ROOT / "models" / "clip-vit-base-patch32"
_HAS_LOCAL = any((LOCAL_CLIP / f).exists() for f in ("pytorch_model.bin", "model.safetensors"))
MODEL_NAME = str(LOCAL_CLIP) if _HAS_LOCAL else "openai/clip-vit-base-patch32"
PROMPT = "a photo of a {} car"


def brand_of(label: str) -> str:
    sep = "__" if "__" in label else "|"
    return label.split(sep)[0]


def load_clip(device: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, proc, torch


def image_features(model, proc, torch, paths: list[Path], device: str, bs: int = 32):
    import numpy as np
    from PIL import Image

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = [Image.open(p).convert("RGB") for p in paths[i : i + bs]]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0)


def text_features(model, proc, torch, brands: list[str], device: str):
    import numpy as np

    with torch.no_grad():
        inputs = proc(text=[PROMPT.format(b) for b in brands], return_tensors="pt", padding=True).to(device)
        f = model.get_text_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()


def collect_split(root: Path, split: str) -> tuple[list[Path], list[str]]:
    paths, labels = [], []
    d = root / split
    for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        for img in cls_dir.glob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(img)
                labels.append(brand_of(cls_dir.name))
    return paths, labels


def realworld_set(dir_: Path) -> tuple[list[Path], list[str]]:
    meta = json.loads((dir_ / "labels.json").read_text(encoding="utf-8"))
    paths = [dir_ / v["file"] for v in meta.values()]
    labels = list(meta.keys())
    return paths, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero-shot", "probe"], default="probe")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "stanford_cars")
    ap.add_argument("--realworld", type=Path, default=ROOT / "test_images" / "realworld")
    ap.add_argument("--export", type=Path, default=None, help="导出探针权重的目录")
    ap.add_argument("--max-train", type=int, default=8000)
    args = ap.parse_args()

    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  mode={args.mode}")

    model, proc, torch = load_clip(device)

    tr_paths, tr_labels = collect_split(args.data, "train")
    va_paths, va_labels = collect_split(args.data, "val")
    rng = random.Random(42)
    if len(tr_paths) > args.max_train:
        idx = rng.sample(range(len(tr_paths)), args.max_train)
        tr_paths = [tr_paths[i] for i in idx]
        tr_labels = [tr_labels[i] for i in idx]
    print(f"[info] train={len(tr_paths)} val={len(va_paths)}")

    brands = sorted(set(tr_labels) | set(va_labels))
    print(f"[info] 品牌数={len(brands)}")

    print("[info] 提取 Stanford 特征 ...")
    Xtr = image_features(model, proc, torch, tr_paths, device)
    ytr = np.array([brands.index(b) for b in tr_labels])
    Xva = image_features(model, proc, torch, va_paths, device)
    yva = np.array([brands.index(b) for b in va_labels])

    rp, rl = realworld_set(args.realworld)
    Xr = image_features(model, proc, torch, rp, device)
    yr = np.array([brands.index(b) if b in brands else -1 for b in rl])

    if args.mode == "zero-shot":
        T = text_features(model, proc, torch, brands, device)
        for name, X, y in (("Stanford val", Xva, yva), ("真实照片", Xr, yr)):
            mask = y >= 0
            pred = (X[mask] @ T.T).argmax(1)
            acc = (pred == y[mask]).mean()
            print(f"  [zero-shot] {name:14s} 品牌准确率 {acc:.2%} ({int((pred == y[mask]).sum())}/{mask.sum()})")
            if name == "真实照片":
                for i, (gt, p) in enumerate(zip(np.array(rl)[mask], pred)):
                    print(f"    {'OK' if brands[p] == gt else '  '} {gt:15s} -> {brands[p]}")
        return

    # 线性探针
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    Xtr_n, Xva_n, Xr_n = normalize(Xtr), normalize(Xva), normalize(Xr)
    clf = LogisticRegression(max_iter=2000, C=10.0, n_jobs=-1)
    clf.fit(Xtr_n, ytr)

    for name, X, y in (("Stanford val", Xva_n, yva), ("真实照片", Xr_n, yr)):
        mask = y >= 0
        pred = clf.predict(X[mask])
        acc = (pred == y[mask]).mean()
        print(f"  [probe] {name:14s} Top-1 {acc:.2%} ({int((pred == y[mask]).sum())}/{mask.sum()})")
        if name == "真实照片":
            for gt, p in zip(np.array(rl)[mask], pred):
                print(f"    {'OK' if brands[p] == gt else '  '} {gt:15s} -> {brands[p]}")

    if args.export:
        out = args.export
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "clip_brand_probe.npz", W=clf.coef_, b=clf.intercept_, classes=np.array(brands))
        print(f"[save] 探针权重 -> {out / 'clip_brand_probe.npz'}")


if __name__ == "__main__":
    main()
