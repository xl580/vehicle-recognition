# 项目设计方案：车牌 / 品牌 / 车型识别系统

> 版本：v1.0 ｜ 日期：2026-11-02

## 1. 项目目标

构建一个端到端的车辆识别系统，输入单张汽车图片（静态照片或视频帧），输出：

1. 车牌号码 + 车牌类型（蓝牌/绿牌/黄牌等）
2. 车辆品牌（Make，如 BMW / 比亚迪）
3. 车型（Model，如 3系 / 秦PLUS）
4. 车身颜色
5. 各项结果的置信度与检测框坐标

## 2. 技术选型及理由

| 模块 | 选型 | 理由 |
|---|---|---|
| 车辆/车牌检测 | YOLOv8s 微调 | 工业事实标准；COCO 预训练直接可检车辆；车牌小目标用 CCPD 微调 |
| 车牌字符识别 | HyperLPR3 | 专为中国车牌设计（蓝/绿/黄/白牌），免训练开箱即用，蓝牌 99%+ |
| 品牌/车型分类 | ResNet50 微调 | CompCars 公开实测品牌 92.8% / 车型 90.1%，路线成熟可复现 |
| 车身颜色 | HSV 统计 + 简单分类头 | 低成本高收益，丰富输出维度 |
| 推理服务 | FastAPI + ONNX Runtime | 异步高性能；ONNX 摆脱训练框架依赖，方便部署 |
| 前端 | 原生 HTML + JS 单页 | 上传 → 展示标注框与结果，够用且零构建成本 |
| 部署 | Docker | 环境可复现，简历必备项 |

**备选方案对比**（为什么不用）：

- PaddleOCR：通用 OCR 强，但车牌专用性弱于 HyperLPR，可作为 fallback。
- 视觉大模型（Qwen-VL / GPT-4o）：零训练出 Demo 快，但细粒度车型不可靠、成本高、无自训练含量，不适合作为简历项目主线。
- YOLOv12/v11：可后期升级，YOLOv8 生态与教程最全，先跑通。

## 3. 数据方案

### 3.1 数据集

| 数据集 | 用途 | 说明 |
|---|---|---|
| CCPD | 车牌检测 + OCR 微调 | 取 CCPD-base 子集 2~5 万张即可；含天气/角度/遮挡子集可做消融 |
| CompCars | 品牌/车型分类 | 用 431 类 make-model 子集（5.2 万张），官方 70/30 划分 |
| 自采数据 | 域适配 | 爬虫或实拍目标场景图片，扩充国产车型类别 |

### 3.2 数据处理流水线

- 车牌：CCPD 文件名即标注（区域+字符），写脚本 `scripts/prepare_ccpd.py` 解析转 YOLO 格式
- 分类：图像按 make-model 建目录，`ImageFolder` 直接加载；bbox 裁剪后训练优于全图
- 增强：RandomResizedCrop / ColorJitter / 光照模拟 / 轻度遮挡（Cutout）

### 3.3 类别策略（重要）

品牌/车型类别不均衡严重，采用：

- 类别数从 431 收敛到 **Top-N 常见车型**（先 50 类验证，再扩）
- 加权采样 + Label Smoothing + Focal Loss 对比实验

## 4. 模型训练方案

### 4.1 车牌检测（YOLOv8）

- 起点：`yolov8s.pt` COCO 预训练
- 数据：CCPD-base 2 万张 train / 2 千张 val
- 超参：imgsz=640, epochs=50, batch=16, lr0=0.01
- 目标指标：mAP@0.5 ≥ 0.95，单帧检测 < 20ms（GPU）

### 4.2 品牌/车型分类（ResNet50）

- 起点：ImageNet 预训练 `resnet50`
- 输入：车辆 bbox 裁剪图 resize 224×224
- 超参：AdamW lr=3e-4, cosine schedule, epochs=30, label smoothing=0.1
- 目标指标：Top-1 ≥ 85%（50 类）/ Top-5 ≥ 97%

### 4.3 评估与 badcase

- 分类混淆矩阵 + 每类 P/R，输出 `docs/eval_report.md`
- 记录 Top 混淆对（如卡宴 vs Macan），针对性补数据

## 5. 系统架构

```
┌────────────┐     ┌──────────────────────────────────┐
│ Web 前端    │────►│ FastAPI  /api/recognize           │
│ 上传/展示   │◄────│ 返回 JSON + 标注框                 │
└────────────┘     └──────┬───────────────────────────┘
                          │ VehicleRecognizer 流水线
                          ▼
        1. YOLOv8 车辆检测（COCO 类别 car/truck/bus）
        2. 裁剪车辆 → YOLOv8 车牌检测 → HyperLPR3 OCR
        3. 裁剪车辆 → ResNet50 品牌/车型 + 颜色
        4. 结果融合（IoU 匹配、置信度加权）
```

## 6. 里程碑计划

| 阶段 | 内容 | 交付物 |
|---|---|---|
| M1 | 环境 + CCPD/CompCars 数据准备 | scripts/ + 数据清单 |
| M2 | YOLOv8 车牌检测训练 | weights/plate_detector.pt + 指标 |
| M3 | ResNet50 分类训练 | weights/vehicle_classifier.pth + 指标 |
| M4 | 流水线整合 + API + 前端 | 可演示系统 |
| M5 | ONNX 导出 + Docker + 压测 | 部署包 + 性能报告 |
| M6 | badcase 分析 + 文档 | eval_report.md + README 定稿 |

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| CompCars 申请周期长 | 先用 Stanford Cars 起步，或爬虫自建小数据集 |
| 国产新能源车覆盖不足 | 自采数据 + 类别分层（品牌必对，车型给 Top-5） |
| 车牌角度大/模糊 | CCPD 增加倾斜子集训练；OCR 前做透视矫正 |
| 部署机器无 GPU | ONNX Runtime CPU 版 + yolov8n 小模型 |

## 8. 实施结果（2026-11 实测）

按上述设计在 RTX 3050 Ti Laptop（4GB）上完成实施，实际落地与原设计的差异和结论：

### 8.1 实测指标

| 模块 | 结果 | 说明 |
|---|---|---|
| 车牌检测（M2） | mAP@0.5 **0.9949**，P 99.89% / R 99.90% | CCPD 9,833 张（base/challenge/tilt/blur/weather 混合），yolov8n，15 epoch，~28min |
| 车型分类（M3） | Top-1 **90.10%** / Top-5 **98.31%** | Stanford Cars 185 类 / 7,376 训练，ResNet50，15 epoch |
| 品牌分类 | 域内 **83.7%**，跨域 42%（12 样本） | CLIP ViT-B/32 特征 + LogisticRegression 探针 |
| 端到端 | GPU **86.6ms/张**（CPU 409ms） | 车辆检测 28ms，OCR 是主要瓶颈 |

### 8.2 与原设计的差异

1. **CompCars → Stanford Cars**：CompCars 完整包 16.5GB（下载不现实），改用 Stanford Cars（HF parquet 镜像，约 1.4GB）。
2. **单分类器 → 双模型**：ResNet50 在棚拍图上验证集 90%，但真实街拍品牌识别仅 33%——棚拍背景捷径学习。增加 CLIP 品牌探针（跨域鲁棒），推理时 CLIP 判品牌 + ResNet50 品牌约束解码。
3. **视图集成**：CLIP 品牌判别用「整图 + 车辆裁剪」特征平均，兼得上下文与主体聚焦（33% → 42%，正确预测置信度 0.2 → 0.95+）。

### 8.3 关键踩坑记录

- OpenCV BGR 与 ToPILImage RGB 通道错位 → 推理时必须先 `cv2.cvtColor`
- 训练端 Resize(256)+CenterCrop 与推理端 Resize(224,224) 不一致 → 两侧严格对齐
- HyperLPR3 车牌类型是 10 类枚举；返回 4 角点坐标需转 xyxy
- ultralytics 默认用 CPU 推理 → 显式 `device=0`
- CCPD 文件名存在逗号与下划线两种 bbox 格式
- ImageFolder train/val 类别不齐 → 取交集对齐后重映射
- Stanford 官方划分存在同车多图泄漏 → 验证集分数偏乐观，需跨域测试集佐证

### 8.4 跨域评估

- 测试集：Wikimedia Commons 实拍 12 品牌（scripts/eval_realworld.py 自动下载与评估）
- 报告：docs/realworld_eval.md（逐张预测 + 置信度）
- 已知局限：类别限于美系市场 1991-2012 车型；国产特供车型不在词表。解决路径：引入网络实拍数据（Car Models 3778，19.3 万张）重训品牌分类器。

## 9. 视频识别与记录功能（v1.1）

- **视频识别**（inference/video.py + POST /api/recognize/video）：按 sample_fps 抽帧（默认 2 帧/秒，跳过过暗帧，上限 max_frames 帧控耗时），逐帧走完整流水线；聚合策略——同车牌号取最高置信度命中（去重），品牌/车型/颜色按命中帧多数投票；为每个车牌保存最佳帧标注图
- **识别记录**（api/db.py，SQLite records/records.db）：图片与视频识别自动入库（时间、来源、文件名、车牌、品牌、车型、颜色、置信度、完整 JSON、标注图路径）；REST 管理接口：列表 / 详情 / 单条删除 / 清空，标注图经 /records/annotated 静态服务
- **前端**：三标签页（图片识别 / 视频识别 / 识别记录），记录页支持来源筛选、缩略图预览、单条删除与清空
- 实测：9 秒 225 帧合成视频，2 帧/秒抽 19 帧，去重命中 2 车牌（置信度 0.992 / 0.9999），总耗时 5.9s（RTX 3050 Ti）
