"""品牌识别：CLIP 特征 gallery + kNN 检索（开放词表）。

问题：线性探针是「闭集」分类器——词表里没有的品牌（比亚迪、吉利、理想…）永远学不出来，
遇到就只能在 49 个错误答案里硬选一个。

方案：把 CLIP 特征当 embedding 建参考库（gallery），推理时算余弦相似度取最近邻。
每个品牌只需几张参考图即可加入，**不需要重训任何模型**，且天然支持 top-k 候选与拒识。

用法：
    # 1. 建库（每品牌最多 30 张，3 视图特征平均）
    python scripts/brand_gallery.py --mode build --per-brand 30

    # 2. 评测（12 张实拍查询图）
    python scripts/brand_gallery.py --mode eval --topk 5

    # 3. 消融：单视图 vs 多视图
    python scripts/brand_gallery.py --mode eval --views full
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

LOCAL_CLIP = ROOT / "models" / "clip-vit-base-patch32"
_HAS_LOCAL = any((LOCAL_CLIP / f).exists() for f in ("pytorch_model.bin", "model.safetensors"))
MODEL_NAME = str(LOCAL_CLIP) if _HAS_LOCAL else "openai/clip-vit-base-patch32"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def brand_of(name: str) -> str:
    sep = "__" if "__" in name else "|"
    return name.split(sep)[0]


def load_backbone(name: str, device: str):
    """返回 (kind, model, proc, torch)。

    clip  ：图文对齐、跨域稳，但细粒度判别一般；
    dinov2：自监督视觉特征，细粒度/实例级检索更强（本项目主要靠它提分）。
    """
    import torch

    if name == "dinov2":
        from transformers import AutoImageProcessor, AutoModel

        mid = "facebook/dinov2-small"
        model = AutoModel.from_pretrained(mid).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(mid)
        return "dinov2", model, proc, torch

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    return "clip", model, proc, torch


def _apply_views(img, views: list[str], yolo=None):
    """从一张原图派生多个视图，提特征后平均，兼得全局与聚焦。

    det：先用 YOLO 找到车身，裁出来再送 CLIP——品牌线索集中在车头/格栅，
    整图里背景（马路、树、其他车）会严重稀释特征。
    """
    from PIL import Image

    out = [img]
    if "flip" in views:
        out.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    if "crop" in views:
        w, h = img.size
        s = 0.75
        out.append(
            img.crop((int(w * (1 - s) / 2), int(h * (1 - s) / 2), int(w * (1 + s) / 2), int(h * (1 + s) / 2)))
        )
    if "det" in views and yolo is not None:
        crop = _detect_crop(yolo, img)
        if crop is not None:
            out.append(crop)
    if "face" in views and yolo is not None:
        crop = _detect_crop(yolo, img)
        if crop is not None:
            # 车身框中心 55%：格栅、大灯、车标所在区域，品牌区分度最高
            w, h = crop.size
            s = 0.55
            out.append(crop.crop((int(w * (1 - s) / 2), int(h * (1 - s) / 2), int(w * (1 + s) / 2), int(h * (1 + s) / 2))))
    return out


def _detect_crop(yolo, img):
    """返回面积最大的车辆裁剪（带 8% padding），检测失败返回 None。"""
    import numpy as np

    arr = np.array(img)[..., ::-1].copy()  # RGB -> BGR，ultralytics 默认按 BGR
    res = yolo.predict(arr, verbose=False, classes=[2, 5, 7], conf=0.25)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[areas.argmax()]
    w, h = img.size
    pw, ph = (x2 - x1) * 0.08, (y2 - y1) * 0.08
    box = (max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(w, int(x2 + pw)), min(h, int(y2 + ph)))
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        return None
    return img.crop(box)


def embed_paths(model, proc, torch, paths: list[Path], device: str, views: list[str], bs: int = 24, yolo=None, kind="clip"):
    """返回 L2 归一化后的特征矩阵 (N, D)；多视图取平均再归一化。"""
    import numpy as np
    from PIL import Image

    feats = []
    with torch.no_grad():
        step = max(1, bs // max(1, len(views)))
        for i in range(0, len(paths), step):
            imgs, counts = [], []
            for p in paths[i : i + step]:
                img = Image.open(p).convert("RGB")
                vs = _apply_views(img, views, yolo)
                imgs.extend(vs)
                counts.append(len(vs))
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            if kind == "dinov2":
                out = model(**inputs)
                f = getattr(out, "pooler_output", None)
                if f is None:
                    f = out.last_hidden_state[:, 0]
            else:
                f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            per, k = [], 0
            for c in counts:
                per.append(f[k : k + c].mean(0))
                k += c
            f = torch.stack(per)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


def collect(root: Path) -> tuple[list[Path], list[str]]:
    """支持两种目录：brand/xxx.jpg（实拍集）与 Brand__Model/xxx.jpg（Stanford）。"""
    paths, labels = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(d.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                paths.append(img)
                labels.append(brand_of(d.name))
    return paths, labels


def realworld_set(dir_: Path) -> tuple[list[Path], list[str]]:
    meta = json.loads((dir_ / "labels.json").read_text(encoding="utf-8"))
    return [dir_ / v["file"] for v in meta.values()], list(meta.keys())


def build(args, model, proc, torch, device, yolo=None, kind='clip') -> None:
    import numpy as np

    paths, labels = collect(args.src)
    rng = np.random.default_rng(42)
    by_brand: dict[str, list[Path]] = {}
    for p, b in zip(paths, labels):
        by_brand.setdefault(b, []).append(p)

    picked_p, picked_b = [], []
    for b, ps in sorted(by_brand.items()):
        idx = rng.permutation(len(ps))[: args.per_brand]
        picked_p += [ps[i] for i in idx]
        picked_b += [b] * len(idx)

    print(f"[info] 品牌 {len(by_brand)} 个，参考图 {len(picked_p)} 张，视图={args.views}")
    feats = embed_paths(model, proc, torch, picked_p, device, args.views, yolo=yolo, kind=kind)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        feats=feats,
        brands=np.array(picked_b),
        paths=np.array([str(p) for p in picked_p]),
        views=np.array(args.views),
    )
    print(f"[save] gallery -> {args.out}  shape={feats.shape}")


def evaluate(args, model, proc, torch, device, yolo=None, kind='clip') -> None:
    import numpy as np

    g = np.load(args.gallery, allow_pickle=True)
    gf, gb = g["feats"], np.array([str(b) for b in g["brands"]])
    brands = sorted(set(gb))

    qp, ql = realworld_set(args.query)
    qf = embed_paths(model, proc, torch, qp, device, args.views, yolo=yolo, kind=kind)

    # 原型法：每品牌取特征均值（对单张噪声更稳）
    protos = np.stack([gf[gb == b].mean(0) for b in brands])
    protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)

    sim_proto = qf @ protos.T
    sim_knn = qf @ gf.T

    def metrics(sim, name, label_of):
        order = np.argsort(-sim, axis=1)
        top1 = [label_of[i] for i in order[:, 0]]
        top3 = [[label_of[i] for i in row[:3]] for row in order]
        acc1 = np.mean([p == t for p, t in zip(top1, ql)])
        acc3 = np.mean([t in c for c, t in zip(top3, ql)])
        print(f"  [{name}] Top-1 {acc1:.1%}  Top-3 {acc3:.1%}")
        return order, top1, acc1, acc3

    print(f"[info] gallery={len(gf)} 张 / {len(brands)} 品牌；查询 {len(qp)} 张；视图={args.views}")
    order_p, top1_p, acc1_p, _ = metrics(sim_proto, "原型中心", brands)
    order_n, top1_n, acc1_n, _ = metrics(sim_knn, "kNN", gb)

    print("\n  逐张结果（kNN）:")
    for i, gt in enumerate(ql):
        cand = [gb[j] for j in order_n[i, :3]]
        mark = "OK" if cand[0] == gt else "--"
        print(f"    {mark} {gt:15s} -> {cand[0]:15s} 候选 {cand}  sim={sim_knn[i, order_n[i,0]]:.3f}")

    # 拒识：低于阈值就交给人工/候选列表，避免「自信地错」
    print("\n  置信度阈值 → 覆盖率/准确率（kNN）:")
    best = sim_knn[np.arange(len(ql)), order_n[:, 0]]
    for t in (0.2, 0.22, 0.24, 0.26, 0.28, 0.30):
        keep = best >= t
        if keep.sum():
            cov = keep.mean()
            acc = np.mean([top1_n[i] == ql[i] for i in np.where(keep)[0]])
            print(f"    sim>={t:.2f}  覆盖 {cov:.0%}  准确 {acc:.0%}")


def leave_one_out(args, model, proc, torch, device, yolo=None, kind='clip') -> None:
    """留出法：每品牌留 holdout 张做查询，其余进库。

    这是最贴近真实用法的评测——"想加一个新品牌，丢几张图进去就行"。
    同时对比旧线性探针（闭集 49 类），后者对词表外品牌必然全错。
    """
    import numpy as np

    paths, labels = collect(args.src)
    if not paths:
        raise SystemExit(f"目录下没有图片：{args.src}")

    # 图片太少的品牌无法同时充当库和查询，直接排除（避免原型为空产生 NaN）
    keep_pairs = [
        (p, l)
        for p, l in zip(paths, labels)
        if sum(1 for x in labels if x == l) >= args.holdout + 2
    ]
    if not keep_pairs:
        raise SystemExit("没有品牌满足 holdout+2 张图片的要求")
    paths = [p for p, _ in keep_pairs]
    labels = [l for _, l in keep_pairs]

    print(f"[info] 共 {len(paths)} 张 / {len(set(labels))} 品牌，视图={args.views}")
    feats = embed_paths(model, proc, torch, paths, device, args.views, yolo=yolo, kind=kind)

    # 每品牌随机留出 holdout 张
    rng = np.random.default_rng(0)
    gal_idx, qry_idx = [], []
    for b in sorted(set(labels)):
        idx = np.array([i for i, x in enumerate(labels) if x == b])
        idx = idx[rng.permutation(len(idx))]
        qry_idx += list(idx[: args.holdout])
        gal_idx += list(idx[args.holdout :])

    gf, gb = feats[gal_idx], np.array([labels[i] for i in gal_idx])
    qf, ql = feats[qry_idx], [labels[i] for i in qry_idx]
    brands = sorted(set(gb))
    print(f"[info] 库 {len(gf)} 张 / {len(brands)} 品牌；查询 {len(qf)} 张")

    protos = np.stack([gf[gb == b].mean(0) for b in brands])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    sim = qf @ protos.T
    order = np.argsort(-sim, axis=1)
    pred = [brands[i] for i in order[:, 0]]
    top3 = [[brands[i] for i in row[:3]] for row in order]

    def report(mask, tag):
        if not mask.any():
            return
        a1 = np.mean([p == t for p, t in zip(np.array(pred)[mask], np.array(ql)[mask])])
        a3 = np.mean([t in c for c, t in zip(np.array(top3)[mask], np.array(ql)[mask])])
        print(f"  {tag:12s} Top-1 {a1:.1%}  Top-3 {a3:.1%}  (n={int(mask.sum())})")

    ql_arr = np.array(ql)
    chinese = {"BYD", "Geely", "Chery", "Wuling", "NIO", "Li Auto", "XPeng", "Changan", "Hongqi", "Haval", "MG", "Roewe"}
    is_cn = np.array([t in chinese for t in ql])
    print("\n  kNN（原型中心）:")
    report(np.ones(len(ql), bool), "全部")
    report(is_cn, "中国品牌")
    report(~is_cn, "海外品牌")

    # 旧方案对照：闭集线性探针
    if args.probe.exists():
        p = np.load(args.probe, allow_pickle=True)
        W, b = p["W"].astype(np.float32), p["b"].astype(np.float32)
        cls = [str(c) for c in p["classes"]]
        logits = qf @ W.T + b
        pred_p = [cls[i] for i in logits.argmax(1)]
        acc_p = np.mean([x == y for x, y in zip(pred_p, ql)])
        acc_cn = np.mean([x == y for x, y, m in zip(pred_p, ql, is_cn) if m])
        print(f"\n  旧线性探针（闭集 49 类）: 全部 Top-1 {acc_p:.1%}  中国品牌 {acc_cn:.1%}")

    best = sim[np.arange(len(ql)), order[:, 0]]
    print("\n  拒识阈值（原型法）:")
    for t in (0.55, 0.60, 0.65, 0.70, 0.75):
        keep = best >= t
        if keep.sum():
            acc = np.mean([pred[i] == ql[i] for i in np.where(keep)[0]])
            print(f"    sim>={t:.2f}  覆盖 {keep.mean():.0%}  准确 {acc:.0%}")

    print("\n  逐张（原型法）:")
    for i, gt in enumerate(ql):
        mark = "OK" if pred[i] == gt else "--"
        print(f"    {mark} {gt:14s} -> {pred[i]:14s} 候选 {top3[i]}  sim={best[i]:.3f}")



PROMPTS = [
    "a photo of a {} car",
    "a {} car",
    "the front of a {} car",
    "a photo of a {} automobile",
]


def zero_shot(args, model, proc, torch, device, yolo=None, kind="clip") -> None:
    """文本零样本 + 与图像检索融合。

    CLIP 的文本塔见过大量「品牌名 + car」的图文对，对品牌名本身敏感；
    图像检索对具体外观敏感。两者错误模式不同，加权融合通常好于任一单路。
    """
    import numpy as np

    paths, labels = collect(args.src)
    keep = [(p, l) for p, l in zip(paths, labels) if sum(1 for x in labels if x == l) >= args.holdout + 2]
    paths = [p for p, _ in keep]
    labels = [l for _, l in keep]

    rng = np.random.default_rng(0)
    qry_idx, gal_idx = [], []
    for b in sorted(set(labels)):
        idx = np.array([i for i, x in enumerate(labels) if x == b])
        idx = idx[rng.permutation(len(idx))]
        qry_idx += list(idx[: args.holdout])
        gal_idx += list(idx[args.holdout :])

    brands = sorted(set(labels))
    feats = embed_paths(model, proc, torch, paths, device, args.views, yolo=yolo, kind=kind)
    qf, ql = feats[qry_idx], [labels[i] for i in qry_idx]
    gf, gb = feats[gal_idx], np.array([labels[i] for i in gal_idx])

    # 文本原型：多 prompt 平均
    with torch.no_grad():
        T = []
        for b in brands:
            inputs = proc(text=[pr.format(b) for pr in PROMPTS], return_tensors="pt", padding=True).to(device)
            f = model.get_text_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            T.append((f.mean(0) / f.mean(0).norm()).cpu().numpy())
        T = np.stack(T)

    # 图像侧：原型中心
    protos = np.stack([gf[gb == b].mean(0) for b in brands])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)

    sim_img = qf @ protos.T
    sim_txt = qf @ T.T

    def z(x):
        return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)

    variants = {"文本零样本": z(sim_txt), "图像检索": z(sim_img), "融合(0.5)": 0.5 * z(sim_txt) + 0.5 * z(sim_img)}
    ql_arr = np.array(ql)
    chinese = {"BYD", "Geely", "Chery", "Wuling", "NIO", "Li Auto", "XPeng", "Changan", "Hongqi", "Haval", "MG", "Roewe"}
    is_cn = np.array([t in chinese for t in ql_arr])

    print(f"\n  查询 {len(ql)} 张 / {len(set(ql))} 品牌，库 {len(gf)} 张")
    for name, sm in variants.items():
        order = np.argsort(-sm, axis=1)
        pred = np.array([brands[i] for i in order[:, 0]])
        top3 = [[brands[i] for i in row[:3]] for row in order]
        a1 = (pred == ql_arr).mean()
        a3 = np.mean([t in c for c, t in zip(top3, ql_arr)])
        cn1 = (pred[is_cn] == ql_arr[is_cn]).mean()
        fg1 = (pred[~is_cn] == ql_arr[~is_cn]).mean()
        print(f"  {name:10s} 全部 Top-1 {a1:.1%} Top-3 {a3:.1%} | 中国 {cn1:.1%} | 海外 {fg1:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "eval", "loo", "zs"], default="eval")
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "stanford_cars" / "train")
    ap.add_argument("--gallery", type=Path, default=ROOT / "weights" / "brand_gallery.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "weights" / "brand_gallery.npz")
    ap.add_argument("--query", type=Path, default=ROOT / "test_images" / "realworld")
    ap.add_argument("--probe", type=Path, default=ROOT / "weights" / "clip_brand_probe.npz")
    ap.add_argument("--backbone", choices=["clip", "dinov2"], default="clip")
    ap.add_argument("--per-brand", type=int, default=30)
    ap.add_argument("--holdout", type=int, default=2)
    ap.add_argument("--views", nargs="+", default=["full", "flip", "crop"])
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    kind, model, proc, torch = load_backbone(args.backbone, device)
    print(f"[info] backbone={args.backbone}")

    yolo = None
    if any(v in args.views for v in ("det", "face")):
        from ultralytics import YOLO

        yolo = YOLO(str(ROOT / "weights" / "yolov8s.pt"))
        print("[info] 已加载 YOLOv8s，用于裁车身视图")

    if args.mode == "build":
        build(args, model, proc, torch, device, yolo, kind)
    elif args.mode == "zs":
        zero_shot(args, model, proc, torch, device, yolo, kind)
    elif args.mode == "loo":
        leave_one_out(args, model, proc, torch, device, yolo, kind)
    else:
        evaluate(args, model, proc, torch, device, yolo, kind)


if __name__ == "__main__":
    main()
