"""从 Wikimedia Commons 抓取品牌街拍图，用于品牌识别评测。

为什么需要：现有 12 张实拍每品牌只有 1 张，无法同时充当参考库和查询集；
且全部是海外品牌，测不出「词表外品牌」（比亚迪、理想等）的真实表现。

用法：
    python scripts/fetch_brand_photos.py --per-brand 6
产出：
    test_images/brand_photos/<Brand>/<n>.jpg
    test_images/brand_photos/sources.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_images" / "brand_photos"
UA = {"User-Agent": "vehicle-recognition-research/1.0 (local academic eval)"}

# 12 个中国品牌（当前模型词表中完全不存在）+ 12 个已覆盖品牌（做对照）
BRANDS = [
    "BYD", "Geely", "Chery", "Wuling", "NIO", "Li Auto",
    "XPeng", "Changan", "Hongqi", "Haval", "MG", "Roewe",
    "Toyota", "Honda", "Volkswagen", "Nissan", "BMW", "Audi",
    "Mercedes-Benz", "Hyundai", "Kia", "Ford", "Chevrolet", "Mazda",
]
SKIP_WORDS = ("logo", "badge", "interior", "engine", "dashboard", "wheel", "seat", "emblem")


def api(params: dict) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def search(brand: str, want: int) -> list[tuple[str, str]]:
    """返回 [(title, thumb_url)]，已过滤车标/内饰类图片。"""
    out: list[tuple[str, str]] = []
    for term in (f"{brand} car front view", f"{brand} automobile front"):
        data = api(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {term}",
                "gsrnamespace": 6,
                "gsrlimit": want * 3,
                "prop": "imageinfo",
                "iiprop": "url|size",
                "iiurlwidth": 900,
                "format": "json",
            }
        )
        for page in (data.get("query", {}).get("pages") or {}).values():
            title = page.get("title", "")
            low = title.lower()
            if any(w in low for w in SKIP_WORDS):
                continue
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("width", 0) < 600:
                continue
            url = info.get("thumburl") or info.get("url")
            if url and all(url != u for _, u in out):
                out.append((title, url))
            if len(out) >= want:
                return out
        if len(out) >= want:
            break
    return out


def download(brand: str, items: list[tuple[str, str]]) -> list[dict]:
    d = OUT / brand.replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, (title, url) in enumerate(items, 1):
        ext = ".jpg"
        path = d / f"{i:02d}{ext}"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                path.write_bytes(r.read())
            saved.append({"file": str(path.relative_to(OUT)).replace("\\", "/"), "title": title})
            print(f"    [{brand}] {i:02d} {title[:70]}")
        except Exception as e:  # 单张失败不影响整体
            print(f"    [{brand}] 跳过：{e}")
        time.sleep(0.3)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-brand", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta: dict[str, list[dict]] = {}
    for brand in BRANDS:
        print(f"[fetch] {brand}")
        items = search(brand, args.per_brand)
        if not items:
            print("    未找到")
            continue
        meta[brand] = download(brand, items)

    (OUT / "sources.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(v) for v in meta.values())
    print(f"\n[done] {len(meta)} 品牌 / {total} 张 -> {OUT}")


if __name__ == "__main__":
    main()
