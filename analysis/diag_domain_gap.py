"""診斷腳本 2：量「訓練域（字型字形）」與「推論域（iPad 4pt 細線）」的線寬差距，
以及分筆轉換後各筆畫在螢幕上的粗細不一致。

模型的 source domain 推測是字型渲染的字形（/synthesize/* 就是直接把 BiauKai
渲染圖餵給同一個模型，app/inference.py:75-86），這裡用容器裡現成的 CJK 字型
（wqy-zenhei，當 標楷體 的替身，標楷體是專有字型、repo 未附）以 inference.py 同樣的
參數（256 畫布、font size 200）渲染整字，量它的線寬分布；再對照 diag_scaling.py
算出的 iPad 筆畫在模型空間的線寬。
第二部分：假設生成器在 256 空間輸出固定粗細 T 的書法筆畫（T 取字型線寬中位數），
算每一筆放回螢幕後的粗細 = T * max(w,h)/256，看同一個字裡各筆的粗細差幾倍。
只需 numpy + Pillow。用法：python analysis/diag_domain_gap.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
CANVAS, FONT_SIZE = 256, 200   # inference.py:26,30


def erosion_width_hist(binary, max_iter=60):
    """每個前景像素到背景的距離（侵蝕次數）直方圖 → 線寬分布（2*d-1）。"""
    b = binary.copy(); dist = np.zeros(binary.shape, int); n = 0
    while b.any() and n < max_iter:
        n += 1; dist[b] = n
        p = np.pad(b, 1); er = p[1:-1, 1:-1].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                er &= p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
        b = er
    return dist[binary]


def render(ch):
    img = Image.new("L", (CANVAS, CANVAS), 255)
    ImageDraw.Draw(img).text((0, 0), ch, 0, font=ImageFont.truetype(FONT, FONT_SIZE))
    return np.asarray(img) < 100


def main():
    out = {}
    print("## 字型字形在 256 畫布的線寬（侵蝕深度最大值*2-1 ≈ 最粗處線寬；中位數 ≈ 典型線寬）")
    print("| 字 | 前景像素 | 最粗線寬 px | 線寬中位數 px |")
    print("|---|---|---|---|")
    meds = []
    for ch in "永和書法一乙了":
        d = erosion_width_hist(render(ch))
        # 典型線寬：以「每個像素所屬的局部最大深度」近似 → 用 90 百分位當最粗、中位數*2-1 當典型
        thick = int(d.max()) * 2 - 1; med = int(np.percentile(d, 75)) * 2 - 1
        meds.append(med)
        print(f"| {ch} | {int(d.size)} | {thick} | {med} |")
        out[ch] = dict(pixels=int(d.size), max_width=thick, typical_width=med)
    T = int(np.median(meds))
    print(f"\n字型典型線寬 T ≈ {T} px（256 空間）；iPad 4pt 線經 fitbest+resize 後是 3~69 px（見 diag_scaling.py）")

    print("\n## 分筆轉換後，同一個字各筆在螢幕上的粗細（假設生成器輸出粗細固定 T）")
    strokes = json.loads(Path(__file__).with_name("diag_scaling.json").read_text())
    print("| 筆畫 | 格寬 pt | bbox 邊長 S | 螢幕粗細 = T*S/256 (pt) |")
    print("|---|---|---|---|")
    rows = {}
    for r in strokes:
        if not r["resample"].startswith("BICUBIC"):
            continue
        S = max(int(v) for v in r["bbox"].split("x"))
        th = T * S / 256
        rows.setdefault(r["cell_pt"], []).append(th)
        print(f"| {r['stroke']} | {r['cell_pt']} | {S} | {th:.1f} |")
    for cell, ths in rows.items():
        print(f"- 格寬 {cell} pt：最粗/最細 = {max(ths):.1f}/{min(ths):.1f} pt，比值 {max(ths)/min(ths):.1f}x（點 vs 長橫）")
    out["T"] = T
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
