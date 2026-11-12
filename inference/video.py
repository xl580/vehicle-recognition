"""视频识别：抽帧 → 逐帧识别 → 聚合汇总。

设计要点：
- 抽帧：默认每秒 2 帧（可调），跳过过暗帧，控制推理次数
- 聚合：同一车牌号取最高置信度的一次命中（去重）；品牌/车型按命中帧多数投票
- 输出：汇总结果 + 每个车牌的最佳帧标注图（保存到 records/annotated/）
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoSummary:
    duration_s: float = 0.0
    frames_total: int = 0
    frames_sampled: int = 0
    frames_with_vehicle: int = 0
    sample_fps: float = 2.0
    plates: list[dict] = field(default_factory=list)   # 去重后的车牌命中
    vehicles: list[dict] = field(default_factory=list)  # 投票后的品牌/车型/颜色
    time_cost_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video": {
                "duration_s": round(self.duration_s, 2),
                "frames_total": self.frames_total,
                "frames_sampled": self.frames_sampled,
                "frames_with_vehicle": self.frames_with_vehicle,
                "sample_fps": self.sample_fps,
            },
            "plates": self.plates,
            "vehicles": self.vehicles,
            "time_cost_ms": {k: round(v, 1) for k, v in self.time_cost_ms.items()},
        }


def _frame_dark(img: np.ndarray, thresh: float = 18.0) -> bool:
    return float(img.mean()) < thresh


def recognize_video(
    rec,
    video_path: str | Path,
    sample_fps: float = 2.0,
    max_frames: int = 300,
    save_annotated_dir: Path | None = None,
    draw_fn=None,
) -> VideoSummary:
    """对视频抽帧识别并聚合。

    rec: inference.pipeline.VehicleRecognizer 实例
    draw_fn(img, result) -> img: 可选的标注绘制函数（如 api.main 里的 draw_annotations）
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps_src if fps_src else 0.0
    step = max(1, int(round(fps_src / max(0.1, sample_fps))))

    summary = VideoSummary(duration_s=duration, frames_total=total, sample_fps=sample_fps)
    t_all = time.perf_counter()

    # plate_text -> {best result, best frame img}
    plate_best: dict[str, dict] = {}
    brand_votes: Counter = Counter()
    model_votes: Counter = Counter()
    color_votes: Counter = Counter()
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if (frame_no - 1) % step != 0:
            continue
        if summary.frames_sampled >= max_frames:
            break
        if _frame_dark(frame):
            continue

        summary.frames_sampled += 1
        result = rec.recognize(frame)
        if result.vehicle_box:
            summary.frames_with_vehicle += 1

        # 车牌：同号取最高置信度
        if result.plate_text:
            cur = {
                "text": result.plate_text,
                "plate_type": result.plate_type,
                "confidence": result.plate_conf,
                "frame": frame_no,
                "time_s": round(frame_no / fps_src, 2) if fps_src else None,
            }
            best = plate_best.get(result.plate_text)
            if best is None or result.plate_conf > best["result"]["confidence"]:
                plate_best[result.plate_text] = {
                    "result": cur,
                    "img": frame.copy(),
                    "res": result,
                }

        # 品牌/车型/颜色：投票
        if result.brand and result.brand != "unknown":
            brand_votes[result.brand] += 1
        if result.model and result.model != "unknown":
            model_votes[f"{result.brand}|{result.model}"] += 1
        if result.color:
            color_votes[result.color] += 1

    cap.release()

    # 车牌列表（按置信度降序），保存最佳帧标注图
    for text, entry in sorted(plate_best.items(), key=lambda kv: -kv[1]["result"]["confidence"]):
        item = dict(entry["result"])
        if save_annotated_dir is not None and draw_fn is not None:
            img = draw_fn(entry["img"], entry["res"])
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                save_annotated_dir.mkdir(parents=True, exist_ok=True)
                safe = "".join(c for c in text if c.isalnum())
                name = f"plate_{safe or 'unknown'}_{item['frame']}.jpg"
                (save_annotated_dir / name).write_bytes(buf.tobytes())
                item["annotated"] = name
        summary.plates.append(item)

    if brand_votes:
        top_brand = brand_votes.most_common(1)[0]
        summary.vehicles.append(
            {"kind": "brand", "value": top_brand[0], "votes": top_brand[1], "frames": summary.frames_sampled}
        )
    if model_votes:
        top_model, votes = model_votes.most_common(1)[0]
        summary.vehicles.append(
            {"kind": "model", "value": top_model.replace("|", " "), "votes": votes, "frames": summary.frames_sampled}
        )
    if color_votes:
        top_color = color_votes.most_common(1)[0]
        summary.vehicles.append(
            {"kind": "color", "value": top_color[0], "votes": top_color[1], "frames": summary.frames_sampled}
        )

    summary.time_cost_ms = {"total": (time.perf_counter() - t_all) * 1000}
    return summary
