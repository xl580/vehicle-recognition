"""ResNet50 品牌/车型细粒度分类训练（CompCars）。

数据组织（ImageFolder 格式，文件夹名 = "品牌|车型"）：

    data/compcars_cls/
    ├── train/
    │   ├── 宝马|3系/xxx.jpg
    │   ├── 比亚迪|秦PLUS/xxx.jpg
    │   └── ...
    └── val/
        └── 同上结构

用法：
    python training/train_classifier.py --data data/compcars_cls --epochs 30 --bs 32

产出：
    weights/vehicle_classifier.pth   模型权重
    weights/label_map.json           类别映射
    docs/eval_report.md              评估报告（混淆矩阵 + 每类指标）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
DOCS = ROOT / "docs"


def align_classes(train_ds, val_ds) -> list[str]:
    """train/val 类别取交集并对齐（某些类在验证集可能无样本）。"""
    common = sorted(set(train_ds.classes) & set(val_ds.classes))
    idx = {c: i for i, c in enumerate(common)}
    for ds in (train_ds, val_ds):
        old = ds.classes
        # 只保留两边都有的类别，其余样本丢弃
        ds.samples = [(p, idx[old[t]]) for p, t in ds.samples if old[t] in idx]
        ds.imgs = ds.samples
        ds.targets = [t for _, t in ds.samples]
        ds.classes = common
        ds.class_to_idx = idx
    return common


def build_loaders(data_dir: Path, bs: int, workers: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)
    classes = align_classes(train_ds, val_ds)
    if len(classes) < 2:
        raise SystemExit("train/val 交集类别不足 2 个，请检查数据目录")

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ds, val_ds, train_ld, val_ld


def build_model(num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    """返回 top1, top5, 混淆矩阵。"""
    model.eval()
    correct1 = correct5 = total = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        top5 = logits.topk(min(5, num_classes), dim=1).indices
        pred = top5[:, 0]
        correct1 += (pred == y).sum().item()
        correct5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
        total += y.size(0)
        for t, p in zip(y.view(-1).cpu(), pred.view(-1).cpu()):
            cm[t, p] += 1
    return correct1 / total, correct5 / total, cm


def write_report(classes: list[str], cm: torch.Tensor, top1: float, top5: float, out: Path) -> None:
    per_class = []
    for i, name in enumerate(classes):
        tp = cm[i, i].item()
        support = cm[i].sum().item()
        pred_total = cm[:, i].sum().item()
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append((name, support, precision, recall, f1))

    worst = sorted(per_class, key=lambda r: r[4])[:10]
    lines = [
        "# 品牌/车型分类评估报告",
        "",
        f"- Top-1 准确率：**{top1:.2%}**",
        f"- Top-5 准确率：**{top5:.2%}**",
        f"- 类别数：{len(classes)}",
        "",
        "## 表现最差的 10 个类别",
        "",
        "| 类别 | 样本数 | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {n} | {s} | {p:.3f} | {r:.3f} | {f:.3f} |" for n, s, p, r, f in worst]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "compcars_cls")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=WEIGHTS / "vehicle_classifier.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    train_ds, val_ds, train_ld, val_ld = build_loaders(args.data, args.bs, args.workers)
    print(f"[info] 类别数={len(train_ds.classes)} train={len(train_ds)} val={len(val_ds)}")

    model = build_model(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 骨干网络用小学习率微调，分类头用全学习率，避免破坏 ImageNet 预训练特征
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    best_top1 = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for i, (x, y) in enumerate(train_ld, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if i % 50 == 0:
                print(f"  epoch {epoch} iter {i}/{len(train_ld)} loss={running / i:.4f}")

        top1, top5, cm = evaluate(model, val_ld, device, len(train_ds.classes))
        scheduler.step()
        print(f"[epoch {epoch}] loss={running / len(train_ld):.4f} top1={top1:.4f} top5={top5:.4f} "
              f"({time.perf_counter() - t0:.1f}s)")

        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), args.out)
            (WEIGHTS / "label_map.json").write_text(
                json.dumps(train_ds.class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            DOCS.mkdir(exist_ok=True)
            write_report(train_ds.classes, cm, top1, top5, DOCS / "eval_report.md")
            print(f"  [save] 新最佳模型 top1={top1:.4f} -> {args.out}")

    print(f"[done] best top1={best_top1:.4f}, 权重：{args.out}")


if __name__ == "__main__":
    main()
