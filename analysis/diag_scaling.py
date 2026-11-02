"""診斷腳本 1：量化「縮放往返」本身（不含模型）造成的誤差。

模擬 iOS ViewController.saveLastStrokeAsImage() 的輸出（整個螢幕大小的白底圖，
4 pt 寬、圓頭的折線），丟進與 app/inference.py 的 fitbest()/preprocess()
完全相同的前處理（bbox → 補成正方形 → resize 256），再模擬 iOS 端把 256x256
結果放回 max(w,h) 正方形 CALayer 的縮放，量：
  * 模型輸入空間裡的線寬（像素）—— 看每種筆畫被放大/縮小後線寬差多少
  * 純縮放往返 IoU（不經模型）—— 縮放本身損失多少
  * 往返後 bbox 中心與原始中心的偏移 —— 定位誤差
  * 去背上色的灰色光暈帶（200<=luma<=240 保留不透明）佔前景比例
fitbest() 的邏輯是從 app/inference.py 逐行複製的（那個模組 import torch，
這個容器沒有 torch，所以不能直接 import），不改任何既有檔案。
只需要 numpy + Pillow。用法：python analysis/diag_scaling.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CANVAS_SIZE = 256
SCREEN = (1024, 1366)  # iPad 點數（iOS 端 fmt.scale = 1，1 pt = 1 px）
LINE_W = 4             # ViewController.swift:71 lineWidth = 4.0

# 測試筆畫集：以 75 pt 練字格（setViewController wordlength 預設 75）
# 與放大到 300 pt 的格子兩種尺寸各畫一次。座標相對格子左上角、單位為格寬比例。
STROKES = {
    "橫":       [(0.10, 0.50), (0.90, 0.50)],
    "豎":       [(0.50, 0.10), (0.50, 0.90)],
    "點":       [(0.48, 0.45), (0.55, 0.55)],
    "撇":       [(0.70, 0.15), (0.55, 0.45), (0.25, 0.85)],
    "橫折":     [(0.15, 0.25), (0.80, 0.25), (0.80, 0.85)],
    "豎彎鉤":   [(0.30, 0.10), (0.30, 0.70), (0.40, 0.85), (0.60, 0.88), (0.85, 0.80), (0.86, 0.70)],
    "橫折鉤":   [(0.15, 0.20), (0.80, 0.20), (0.80, 0.80), (0.65, 0.72)],
    "長橫(整行)": [(0.02, 0.50), (0.98, 0.50)],
}
CELL_SIZES = [75, 150, 300]


def rasterize_like_ios(points, cell, origin=(300, 400)):
    """模擬 UIGraphicsImageRenderer：白底、4pt 黑線、圓頭圓角。"""
    img = Image.new("L", SCREEN, 255)
    d = ImageDraw.Draw(img)
    pts = [(origin[0] + x * cell, origin[1] + y * cell) for x, y in points]
    d.line(pts, fill=0, width=LINE_W, joint="curve")
    r = LINE_W / 2
    for p in (pts[0], pts[-1]):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=0)
    return img


def fitbest(image):
    """逐行複製自 app/inference.py:89-132"""
    gray = image.convert("L")
    npimage = np.asarray(gray)
    rows, cols = np.where(npimage < 100)
    min_x, max_x = int(rows.min()), int(rows.max())
    min_y, max_y = int(cols.min()), int(cols.max())
    cropped = gray.crop((min_y - 1, min_x - 1, max_y + 1, max_x + 1))
    width = max_y - min_y + 2
    height = max_x - min_x + 2
    canvas_size = max(width, height)
    canvas = Image.new("L", (canvas_size, canvas_size), 255)
    if width > height:
        canvas.paste(cropped, (0, (canvas_size - height) // 2))
    else:
        canvas.paste(cropped, ((canvas_size - width) // 2, 0))
    meta = dict(canvas_size=canvas_size, min_x=min_x, min_y=min_y, width=width, height=height)
    return canvas, meta


def line_width_px(binary):
    """以反覆 3x3 侵蝕估線寬：侵蝕到空為止的次數 ≈ 半線寬。純 numpy。"""
    b = binary.copy()
    n = 0
    while b.any():
        p = np.pad(b, 1)
        er = p[1:-1, 1:-1]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                er = er & p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
        b = er
        n += 1
    return 2 * n - 1  # 近似：中心線一列 + 兩側各 n-1


def iou(a, b):
    return float((a & b).sum()) / max(1, float((a | b).sum()))


def bbox_center(binary):
    rows, cols = np.where(binary)
    return ((cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2)


def main():
    out_rows = []
    for name, pts in STROKES.items():
        for cell in CELL_SIZES:
            img = rasterize_like_ios(pts, cell)
            orig = np.asarray(img) < 100
            canvas, meta = fitbest(img)
            for method_name, method in [("BICUBIC(現況預設)", Image.BICUBIC), ("NEAREST", Image.NEAREST), ("LANCZOS", Image.LANCZOS)]:
                small = canvas.resize((CANVAS_SIZE, CANVAS_SIZE), method)   # inference.py:165
                small_bin = np.asarray(small) < 100
                lw_model = line_width_px(small_bin)
                # 往返：模擬 iOS CALayer 把 256x256 放回 max(w,h) 正方形
                S = meta["canvas_size"]
                back = small.resize((S, S), method)
                back_np = np.asarray(back)
                back_bin = back_np < 100
                # 貼回原圖座標（x_position/y_position 公式，inference.py:180-183）
                full = np.zeros(orig.shape, dtype=bool)
                cx = meta["min_y"] + meta["width"] / 2
                cy = meta["min_x"] + meta["height"] / 2
                x0 = int(round(cx - S / 2)); y0 = int(round(cy - S / 2))
                xs = slice(max(0, x0), min(orig.shape[1], x0 + S)); ys = slice(max(0, y0), min(orig.shape[0], y0 + S))
                full[ys, xs] = back_bin[: ys.stop - ys.start, : xs.stop - xs.start]
                rt_iou = iou(orig, full)
                oc = bbox_center(orig); rc = bbox_center(full)
                # 對齊中心後的 IoU：扣掉定位偏移，只剩重採樣本身的損失
                dx = int(round(oc[0] - rc[0])); dy = int(round(oc[1] - rc[1]))
                aligned = np.roll(np.roll(full, dy, axis=0), dx, axis=1)
                al_iou = iou(orig, aligned)
                # 灰色光暈帶：_tint_strokes() 把 200<=luma<=240 的像素原樣保留（不透明）
                luma = back_np.astype(np.float32)
                halo = ((luma >= 200) & (luma <= 240)).sum() / max(1, (luma < 200).sum())
                row = dict(stroke=name, cell_pt=cell, bbox=f'{meta["width"]}x{meta["height"]}',
                           scale_to_256=round(CANVAS_SIZE / S, 2), resample=method_name,
                           line_w_model_px=lw_model, roundtrip_iou=round(rt_iou, 3), aligned_iou=round(al_iou, 3),
                           center_err_px=(float(round(rc[0] - oc[0], 1)), float(round(rc[1] - oc[1], 1))),
                           halo_ratio=round(float(halo), 2))
                out_rows.append(row)
    # 印成 markdown 表
    hdr = ["stroke", "cell_pt", "bbox", "scale_to_256", "resample", "line_w_model_px", "roundtrip_iou", "aligned_iou", "center_err_px", "halo_ratio"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in out_rows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    Path(__file__).with_suffix(".json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=1))
    # 摘要：以 BICUBIC 看線寬極值
    bic = [r for r in out_rows if r["resample"].startswith("BICUBIC")]
    lws = [r["line_w_model_px"] for r in bic]
    print(f"\n[摘要] BICUBIC：模型輸入空間線寬 min={min(lws)}px max={max(lws)}px（比值 {max(lws)/max(1,min(lws)):.1f}x）")
    print(f"[摘要] BICUBIC：往返 IoU min={min(r['roundtrip_iou'] for r in bic):.3f} max={max(r['roundtrip_iou'] for r in bic):.3f}")
    print(f"[摘要] BICUBIC：對齊中心後 IoU min={min(r['aligned_iou'] for r in bic):.3f} max={max(r['aligned_iou'] for r in bic):.3f}")
    print(f"[摘要] BICUBIC：光暈帶比例 min={min(r['halo_ratio'] for r in bic):.2f} max={max(r['halo_ratio'] for r in bic):.2f}")


if __name__ == "__main__":
    main()
