"""识别记录存储（SQLite）。

表结构 records：
    id            自增主键
    ts            识别时间（ISO 字符串，本地时区）
    source_type   image / video
    filename      原始文件名
    plate_text    车牌号（视频来源为聚合后的最优车牌）
    plate_type    车牌类型
    plate_conf    车牌置信度
    brand         品牌
    model         车型
    color         车身颜色
    vehicle_conf  车辆分类置信度
    time_cost_ms  各阶段耗时（JSON 字符串）
    detail        完整结果 JSON（视频来源为汇总结果）
    annotated     标注图相对路径（保存于 records/ 目录，可空）
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "records" / "records.db"
ANNOTATED_DIR = ROOT / "records" / "annotated"

_COLUMNS = (
    "id, ts, source_type, filename, plate_text, plate_type, plate_conf, "
    "brand, model, color, vehicle_conf, time_cost_ms, detail, annotated"
)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                plate_text TEXT DEFAULT '',
                plate_type TEXT DEFAULT '',
                plate_conf REAL DEFAULT 0,
                brand TEXT DEFAULT '',
                model TEXT DEFAULT '',
                color TEXT DEFAULT '',
                vehicle_conf REAL DEFAULT 0,
                time_cost_ms TEXT DEFAULT '{}',
                detail TEXT DEFAULT '{}',
                annotated TEXT DEFAULT ''
            )
            """
        )


def insert_record(result: dict, source_type: str, filename: str, annotated: str = "") -> int:
    """把 recognize() 的 to_dict() 结果写入一条记录，返回记录 id。"""
    plate = result.get("plate", {}) or {}
    vehicle = result.get("vehicle", {}) or {}
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO records (ts, source_type, filename, plate_text, plate_type, plate_conf,"
            " brand, model, color, vehicle_conf, time_cost_ms, detail, annotated)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                source_type,
                filename,
                plate.get("text", ""),
                plate.get("plate_type", ""),
                float(plate.get("confidence", 0) or 0),
                vehicle.get("brand", ""),
                vehicle.get("model", ""),
                vehicle.get("color", ""),
                float(vehicle.get("confidence", 0) or 0),
                json.dumps(result.get("time_cost_ms", {}), ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                annotated,
            ),
        )
        return int(cur.lastrowid)


def list_records(limit: int = 100, source_type: str | None = None) -> list[dict]:
    q = f"SELECT {_COLUMNS} FROM records"
    params: list = []
    if source_type in ("image", "video"):
        q += " WHERE source_type = ?"
        params.append(source_type)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(q, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["time_cost_ms"] = json.loads(d.get("time_cost_ms") or "{}")
        d.pop("detail", None)  # 列表页不带完整 JSON，减小体积
        out.append(d)
    return out


def get_record(record_id: int) -> dict | None:
    with _connect() as conn:
        r = conn.execute(f"SELECT {_COLUMNS} FROM records WHERE id = ?", (record_id,)).fetchone()
    if r is None:
        return None
    d = dict(r)
    d["time_cost_ms"] = json.loads(d.get("time_cost_ms") or "{}")
    d["detail"] = json.loads(d.get("detail") or "{}")
    return d


def delete_record(record_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        return cur.rowcount > 0


def clear_records() -> int:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM records")
        return cur.rowcount
