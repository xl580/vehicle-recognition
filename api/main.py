"""FastAPI 推理服务。

启动：
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

接口：
    GET  /api/health             健康检查
    POST /api/recognize          上传图片，返回识别 JSON（自动写入记录）
    POST /api/recognize/annotated 返回带标注框的 JPEG
    POST /api/recognize/video    上传视频，抽帧识别 + 聚合汇总（自动写入记录）
    GET  /api/records            识别记录列表（?limit=100&source_type=image|video）
    GET  /api/records/{id}       记录详情（含完整 JSON）
    DELETE /api/records/{id}     删除单条记录
    DELETE /api/records          清空记录
    GET  /records/annotated/x.jpg 标注图静态服务
"""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import db  # noqa: E402
from inference.pipeline import VehicleRecognizer  # noqa: E402
from inference.video import recognize_video  # noqa: E402

recognizer: VehicleRecognizer | None = None
TMP_DIR = ROOT / "records" / "tmp"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    print("[startup] 正在加载模型...")
    t = time.perf_counter()
    db.init_db()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = VehicleRecognizer()
    print(f"[startup] 模型加载完成，耗时 {time.perf_counter() - t:.2f}s")
    yield
    recognizer = None


app = FastAPI(title="Vehicle Recognition API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

ANNOTATED_DIR = ROOT / "records" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/records/annotated", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated")


@app.get("/")
async def index():
    page = WEB_DIR / "index.html"
    return FileResponse(str(page)) if page.exists() else {"msg": "Vehicle Recognition API"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if recognizer is not None else "loading",
        "plate_detector": bool(recognizer and recognizer.det_plate),
        "ocr": recognizer.ocr is not None if recognizer else False,
        "classifier": bool(recognizer and recognizer.classifier),
        "brand_clf": bool(recognizer and recognizer.brand_clf),
        "records_db": db.DB_PATH.exists(),
    }


# ----------------------------------------------------------------------
def draw_annotations(img: np.ndarray, result) -> np.ndarray:
    """在图上画车辆框（绿）、车牌框（红）与识别文本。"""
    d = result.to_dict()
    if d["vehicle"]["box"]:
        x1, y1, x2, y2 = d["vehicle"]["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (58, 138, 55), 2)
        brand = f'{d["vehicle"]["brand"]} {d["vehicle"]["model"]}'.strip()
        if brand:
            cv2.putText(img, brand, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (58, 138, 55), 2)
    if d["plate"]["box"]:
        x1, y1, x2, y2 = d["plate"]["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (179, 74, 83), 2)
        if d["plate"]["text"]:
            cv2.putText(
                img, d["plate"]["text"], (x1, min(img.shape[0] - 8, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (179, 74, 83), 2
            )
    return img


def save_annotated(img: np.ndarray, result) -> str:
    """保存标注图到 records/annotated/，返回文件名。"""
    img = draw_annotations(img.copy(), result)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
    (ANNOTATED_DIR / name).write_bytes(buf.tobytes())
    return name


# ----------------------------------------------------------------------
@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...)):
    if recognizer is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片超过 15MB 限制")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片，请上传 jpg/png")

    try:
        result = recognizer.recognize(img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"识别失败: {exc}") from exc

    annotated = save_annotated(img, result)
    record_id = db.insert_record(result.to_dict(), "image", file.filename or "image.jpg", annotated)

    return {"filename": file.filename, "record_id": record_id, "annotated": f"/records/annotated/{annotated}" if annotated else None, **result.to_dict()}


@app.post("/api/recognize/annotated")
async def recognize_annotated(file: UploadFile = File(...)):
    """返回带标注框的 JPEG 图片，便于前端直接展示。"""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片")

    result = recognizer.recognize(img)
    img = draw_annotations(img, result)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="结果图编码失败")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ----------------------------------------------------------------------
@app.post("/api/recognize/video")
async def recognize_video_api(
    file: UploadFile = File(...),
    sample_fps: float = Query(2.0, ge=0.5, le=10.0, description="抽帧频率（帧/秒）"),
    max_frames: int = Query(150, ge=1, le=600, description="最多识别的帧数"),
):
    """上传视频（mp4/avi/mov），抽帧识别并聚合汇总；每个命中车牌写入一条记录。"""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(status_code=400, detail=f"不支持的视频格式 {suffix}")

    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="视频超过 200MB 限制")

    tmp_path = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    tmp_path.write_bytes(data)

    try:
        summary = recognize_video(
            recognizer,
            tmp_path,
            sample_fps=sample_fps,
            max_frames=max_frames,
            save_annotated_dir=ANNOTATED_DIR,
            draw_fn=draw_annotations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"视频识别失败: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as e:  # 环境的安全删除策略可能拦截，留待下次清理
            print(f"[warn] 临时视频删除失败（忽略）: {e}")

    d = summary.to_dict()
    filename = file.filename or "video.mp4"

    # 每个命中车牌写一条记录（无车牌则写一条视频汇总记录）
    if d["plates"]:
        for plate in d["plates"]:
            row = {
                "plate": {"text": plate["text"], "plate_type": plate.get("plate_type", ""),
                          "confidence": plate.get("confidence", 0), "box": []},
                "vehicle": {"brand": (d["vehicles"][0]["value"] if d["vehicles"] else ""),
                            "model": next((v["value"] for v in d["vehicles"] if v["kind"] == "model"), ""),
                            "color": next((v["value"] for v in d["vehicles"] if v["kind"] == "color"), ""),
                            "confidence": (d["vehicles"][0]["value"] and plate.get("confidence", 0)) or 0, "box": []},
                "time_cost_ms": d["time_cost_ms"],
            }
            db.insert_record(row, "video", filename, plate.get("annotated", ""))
    else:
        db.insert_record({"plate": {}, "vehicle": {}, "time_cost_ms": d["time_cost_ms"]}, "video", filename)

    return {"filename": filename, "records_written": max(1, len(d["plates"])), **d}


# ----------------------------------------------------------------------
@app.get("/api/records")
async def records_list(limit: int = Query(100, ge=1, le=500), source_type: str | None = Query(None)):
    if source_type and source_type not in ("image", "video"):
        raise HTTPException(status_code=400, detail="source_type 只能是 image 或 video")
    return {"total_limit": limit, "records": db.list_records(limit=limit, source_type=source_type)}


@app.get("/api/records/{record_id}")
async def records_detail(record_id: int):
    r = db.get_record(record_id)
    if r is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    return r


@app.delete("/api/records/{record_id}")
async def records_delete(record_id: int):
    if not db.delete_record(record_id):
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"deleted": record_id}


@app.delete("/api/records")
async def records_clear():
    return {"deleted": db.clear_records()}
