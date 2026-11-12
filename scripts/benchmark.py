"""推理性能基准测试：对比 CPU / GPU 耗时，输出可直接写进简历的数据。

用法：
    python scripts/benchmark.py --image test_images/shanghai_nev_green.jpg --repeat 10
    python scripts/benchmark.py --image xxx.jpg --device cpu
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.pipeline import VehicleRecognizer  # noqa: E402


def bench(rec: VehicleRecognizer, image: str, repeat: int) -> dict[str, float]:
    # 预热：首帧包含 CUDA 上下文初始化 / 模型加载，不计入统计
    rec.recognize(image)

    detect, ocr, total = [], [], []
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = rec.recognize(image)
        total.append((time.perf_counter() - t0) * 1000)
        detect.append(res.time_cost_ms.get("detection", 0.0))
        ocr.append(res.time_cost_ms.get("ocr", 0.0))

    return {
        "detect_mean": statistics.mean(detect),
        "ocr_mean": statistics.mean(ocr),
        "total_mean": statistics.mean(total),
        "total_p95": sorted(total)[max(0, int(len(total) * 0.95) - 1)],
        "fps": 1000 / statistics.mean(total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "test_images" / "shanghai_nev_green.jpg"))
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    args = ap.parse_args()

    if not Path(args.image).exists():
        raise SystemExit(f"图片不存在：{args.image}")

    print(f"图片: {Path(args.image).name} | 重复 {args.repeat} 次（含 1 次预热）\n")
    results = {}
    for tag, dev in (("GPU (RTX 3050 Ti)", 0), ("CPU", "cpu")):
        if dev == 0:
            try:
                import torch

                if not torch.cuda.is_available():
                    print("GPU 不可用，跳过")
                    continue
            except ImportError:
                continue
        if args.device == "cpu" and dev == 0:
            continue
        if args.device == "cuda" and dev != 0:
            continue

        rec = VehicleRecognizer(device=dev)
        r = bench(rec, args.image, args.repeat)
        results[tag] = r
        print(f"[{tag}]")
        print(f"  检测  {r['detect_mean']:7.1f} ms")
        print(f"  OCR   {r['ocr_mean']:7.1f} ms")
        print(f"  单图  {r['total_mean']:7.1f} ms (P95 {r['total_p95']:.1f} ms)")
        print(f"  FPS   {r['fps']:7.1f}")
        print()

    if len(results) == 2:
        gpu, cpu = results["GPU (RTX 3050 Ti)"], results["CPU"]
        print(f"加速比: {cpu['total_mean'] / gpu['total_mean']:.1f}x（单图 {cpu['total_mean']:.0f}ms -> {gpu['total_mean']:.0f}ms）")


if __name__ == "__main__":
    main()
