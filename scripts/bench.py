"""量測 /predict/word、/predict/stroke 的端對端延遲（平均 / p95）。

用法：
    python scripts/bench.py                     # 預設打 127.0.0.1:8000，各端點各 100 次
    BENCH_N=20 python scripts/bench.py           # 改跑的次數
    BENCH_BASE_URL=http://136.67.17.108:8080 python scripts/bench.py   # 量經 frp 中繼的延遲

注意：這支腳本會在短時間內連續打 N 次 /predict/*，一定會撞到
app/config.py 裡的 RATE_LIMIT_PREDICT（預設 10/minute）。跑之前
先把 .env 的 RATE_LIMIT_PREDICT 臨時調高（例如 1000/minute）、
docker compose up -d 重啟套用，量完再調回來——不要把這麼寬鬆的
限制留在正式環境。
"""
import base64
import os
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

BASE_URL = os.getenv("BENCH_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")
N = int(os.getenv("BENCH_N", "100"))

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def make_sample_image() -> bytes:
    """跟 scripts/smoke_test.py 一樣，用字型畫一個「永」字當測試輸入，
    不需要真的準備一張手寫掃描圖。"""
    font = ImageFont.truetype(str(ROOT / "fonts" / "BiauKai.ttf"), size=200)
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "永", (0, 0, 0), font=font)
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bench_endpoint(path: str, sample: bytes, n: int) -> None:
    times = []
    errors = 0
    for _ in range(n):
        start = time.perf_counter()
        r = requests.post(
            f"{BASE_URL}{path}",
            files={"file": ("sample.png", sample, "image/png")},
            headers=HEADERS,
        )
        elapsed = (time.perf_counter() - start) * 1000
        if r.status_code != 200:
            errors += 1
            print(f"  [!] {path} 回 {r.status_code}: {r.text[:200]}")
            continue
        times.append(elapsed)

    if not times:
        print(f"{path}: 全部失敗（{errors}/{n}），跳過統計")
        return

    times.sort()
    p95_idx = max(0, int(len(times) * 0.95) - 1)
    print(f"{path}  (n={len(times)}, 失敗 {errors})")
    print(f"  平均: {statistics.mean(times):.1f} ms")
    print(f"  中位數: {statistics.median(times):.1f} ms")
    print(f"  p95 : {times[p95_idx]:.1f} ms")
    print(f"  min/max: {times[0]:.1f} / {times[-1]:.1f} ms")


def main() -> None:
    print(f"目標: {BASE_URL}  次數: {N}  帶 API Key: {'是' if API_KEY else '否'}")
    sample = make_sample_image()

    for path in ["/predict/word", "/predict/stroke"]:
        bench_endpoint(path, sample, N)
        print()


if __name__ == "__main__":
    main()
