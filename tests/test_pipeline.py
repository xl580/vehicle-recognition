"""端到端接口测试。

运行：
    pytest tests/ -v

无模型权重时，部分用例会自动 skip（流水线支持降级）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.pipeline import VehicleRecognizer, estimate_color  # noqa: E402


@pytest.fixture(scope="module")
def recognizer():
    try:
        return VehicleRecognizer()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"模型加载失败（缺少依赖或权重）: {exc}")


def _dummy_image(w=640, h=480):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[200:400, 150:500] = (40, 40, 40)
    return img


def test_estimate_color_red():
    img = np.full((100, 100, 3), (0, 0, 220), dtype=np.uint8)  # BGR 红
    assert estimate_color(img) == "红色"


def test_estimate_color_white():
    img = np.full((100, 100, 3), 235, dtype=np.uint8)
    assert estimate_color(img) in ("白色", "黑色/深灰")


def test_recognize_returns_structure(recognizer):
    res = recognizer.recognize(_dummy_image())
    data = res.to_dict()
    assert set(data) == {"plate", "vehicle", "time_cost_ms"}
    assert isinstance(data["plate"]["text"], str)
    assert isinstance(data["vehicle"]["brand"], str)


def test_recognize_invalid_input(tmp_path, recognizer):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        recognizer.recognize(str(bad))


def test_api_health():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "loading")
