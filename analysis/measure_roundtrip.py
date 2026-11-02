#!/usr/bin/env python
"""量測「前處理縮放 -> (模型略過) -> 後處理縮放回去」的往返誤差。

只讀取 smart-calligraphy-api/app/inference.py（用來做 AST 一致性檢查），
不修改任何既有檔案；所有輸出只寫進本資料夾 (analysis/) 與 analysis/stages/。

用法（Windows 與 Linux 相同）：
    python analysis/measure_roundtrip.py [path/to/smart-calligraphy-api]

只依賴 numpy + Pillow，不需要 torch。因為沒有 torch，模型這一步以「恆等」代替
（fake_B := 輸入 tensor），因此這裡量到的只有 fitbest/resize/tensor 量化/去背上色/
貼回 這幾步造成的誤差，不含模型本身的生成誤差。

忠實度說明：
  * fitbest()、_tint_strokes() 是從 inference.py 逐字複製，main() 會用 AST 比對確認一致。
  * ToTensor / Normalize(0.5,0.5) / (x*0.5+0.5) / ToPILImage 以 numpy float32 重現。
    ToPILImage 對 float tensor 是 mul(255).byte()（截斷而非四捨五入）——這是依
    torchvision 的已知行為重現，本機未安裝 torchvision，無法直接驗證。
  * iOS 端「縮放回原尺寸」是 CALayer 把 256x256 圖片放進 bounds=(side,side) 時由
    Core Animation 完成，沒有 Swift 程式碼做縮放；這裡用 PIL BILINEAR / BICUBIC
    近似（兩者都報），不是 iOS 的真實濾波器。
"""
import ast
import json
import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as PIL_VERSION

HERE = Path(__file__).resolve().parent
STAGES = HERE / "stages"
CANVAS_SIZE = 256
SS = 4  # 合成筆跡時的超取樣倍率（模擬 iOS 抗鋸齒）
IOS_CANVAS = (768, 1024)  # 假設的 iPad 點數尺寸（舊 server 的 outputimg 用 768x1024，[推測]）
WEB_CANVAS = (256, 256)

# --------------------------------------------------------------------------
# 逐字複製自 smart-calligraphy-api/app/inference.py（main() 內有 AST 一致性檢查）
# --------------------------------------------------------------------------


def fitbest(image: Image.Image) -> Tuple[Image.Image, dict]:
    """從原圖裡找出筆劃的邊界框，裁切後置中貼進一個正方形畫布。"""
    gray = image.convert("L")
    npimage = np.asarray(gray)
    (imgwidth, imgheight) = gray.size

    rows, cols = np.where(npimage < 100)
    if rows.size == 0:
        raise ValueError("圖片裡偵測不到筆劃（找不到任何深色像素）")

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

    meta = {
        "canvas_size": canvas_size,
        "min_x": min_x,
        "min_y": min_y,
        "width": width,
        "height": height,
        "orig_width": imgwidth,
        "orig_height": imgheight,
    }
    return canvas, meta


def _tint_strokes(image: Image.Image, color: Tuple[int, int, int, int]) -> Image.Image:
    """白色背景去背成透明、深色筆劃塗成指定顏色，中間的反鋸齒邊緣像素原樣保留。"""
    rgba = np.array(image.convert("RGBA"))
    luma = (
        rgba[..., 0].astype(np.float32) * 30
        + rgba[..., 1].astype(np.float32) * 59
        + rgba[..., 2].astype(np.float32) * 11
        + 50
    ) / 100

    out = rgba.copy()
    out[luma > 240, 3] = 0
    out[luma < 200] = np.array(color, dtype=np.uint8)
    return Image.fromarray(out, mode="RGBA")


def verify_verbatim(inference_py: Path) -> dict:
    """比對本檔的 fitbest/_tint_strokes 與 inference.py 的 AST（忽略 docstring）。"""

    def strip_doc(fn: ast.FunctionDef) -> str:
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        fn2 = ast.FunctionDef(name=fn.name, args=fn.args, body=body, decorator_list=fn.decorator_list,
                              returns=fn.returns, type_comment=None, type_params=[])
        return ast.dump(fn2, include_attributes=False)

    def funcs(tree):
        return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    mine = funcs(ast.parse(Path(__file__).read_text(encoding="utf-8")))
    theirs = funcs(ast.parse(inference_py.read_text(encoding="utf-8")))
    return {name: strip_doc(mine[name]) == strip_doc(theirs[name]) for name in ("fitbest", "_tint_strokes")}


# --------------------------------------------------------------------------
# 管線（略過模型）
# --------------------------------------------------------------------------


def tensor_roundtrip(img256: Image.Image) -> Image.Image:
    """ToTensor -> Normalize(0.5,0.5) -> [模型恆等] -> *0.5+0.5 -> clamp -> ToPILImage。"""
    f32 = np.float32
    a = np.asarray(img256.convert("L"), dtype=np.uint8).astype(f32) / f32(255.0)  # ToTensor
    n = (a - f32(0.5)) / f32(0.5)                                                # Normalize
    d = np.clip(n * f32(0.5) + f32(0.5), 0, 1)                                    # _finalize
    b = (d * f32(255.0)).astype(np.uint8)                                         # ToPILImage: mul(255).byte()
    return Image.fromarray(b, mode="L")


def composite_full(layer: Image.Image, x_pos: float, y_pos: float, size_wh):
    """模擬 iOS：layer.bounds=(side,side)，layer.position=(x_pos,y_pos)，anchor 預設 0.5。
    座標四捨五入到整數像素（實機有 0.5pt 以內的次像素差，[推測]）。"""
    side = layer.size[0]
    x0 = math.floor(x_pos - side / 2 + 0.5)
    y0 = math.floor(y_pos - side / 2 + 0.5)
    base = Image.new("RGBA", size_wh, (255, 255, 255, 255))
    base.paste(layer, (x0, y0), layer)
    return base.convert("L"), (x0, y0)


def over_white(rgba: Image.Image) -> Image.Image:
    base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    base.alpha_composite(rgba)
    return base.convert("L")


def run_pipeline(img: Image.Image, filt):
    square, meta = fitbest(img)
    side = square.size[0]
    inp = square.resize((CANVAS_SIZE, CANVAS_SIZE))  # 與 preprocess() 相同：未指定 resample
    rt = tensor_roundtrip(inp)
    tinted = _tint_strokes(rt, (0, 0, 0, 255))
    out_side = max(meta["width"], meta["height"])
    x_pos = meta["min_y"] + meta["width"] / 2
    y_pos = meta["min_x"] + meta["height"] / 2
    layer = tinted.resize((out_side, out_side), filt)
    recon_full, origin = composite_full(layer, x_pos, y_pos, img.size)
    return dict(square=square, meta=meta, side=side, inp=inp, rt=rt, tinted=tinted, layer=layer,
                x_pos=x_pos, y_pos=y_pos, recon_full=recon_full, origin=origin)


# --------------------------------------------------------------------------
# 合成筆跡（iOS: 白底黑線 lineWidth=4, round cap/join；Web: 256x256, lineWidth=8）
# --------------------------------------------------------------------------


def render_stroke(size_wh, pts, width):
    W, H = size_wh
    big = Image.new("L", (W * SS, H * SS), 255)
    d = ImageDraw.Draw(big)
    r = width * SS / 2
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        d.line([(x0 * SS, y0 * SS), (x1 * SS, y1 * SS)], fill=0, width=int(round(width * SS)))
    for x, y in pts:
        d.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=0)
    return big.resize((W, H), Image.BOX)


def arc(cx, cy, r, a0, a1, step_deg=4):
    n = max(2, int(abs(a1 - a0) / step_deg))
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def build_samples():
    S = []

    def add(sid, desc, size, pts, width, probes):
        S.append(dict(id=sid, desc=desc, size=size, pts=pts, width=width, probes=probes))

    # probes: ("v", x, (y0,y1)) = 沿垂直線 x 量過筆劃的等效寬度；("h", y, (x0,x1)) 同理
    add("I01_vertical_long", "細長豎 4x300", IOS_CANVAS, [(384, 300), (384, 600)], 4, [("h", 450, (370, 400))])
    add("I02_horizontal_long", "扁長橫 300x4", IOS_CANVAS, [(234, 500), (534, 500)], 4, [("v", 384, (485, 515))])
    add("I03_vertical_grid", "格內尺度豎 4x60", IOS_CANVAS, [(384, 300), (384, 360)], 4, [("h", 330, (370, 400))])
    add("I04_horizontal_grid", "格內尺度橫 60x4", IOS_CANVAS, [(354, 500), (414, 500)], 4, [("v", 384, (485, 515))])
    add("I05_hengzhe_large", "橫折 200+200", IOS_CANVAS, [(200, 300), (400, 300), (400, 500)], 4,
        [("v", 300, (285, 315)), ("h", 400, (385, 415))])
    add("I06_hengzhe_grid", "橫折 格內 55+55", IOS_CANVAS, [(300, 300), (355, 300), (355, 355)], 4,
        [("v", 327, (285, 315)), ("h", 330, (340, 370))])
    # 豎彎鉤：豎 90 -> 圓弧(r=25) -> 橫 40 -> 鉤(向上)
    stem_end = (300, 400)
    ar = arc(325, 400, 25, 180, 90)  # 從 (300,400) 向右下彎到 (325,425)
    sw = [(300, 310), stem_end] + ar[1:] + [(385, 425), (395, 400)]
    add("I07_shuwangou", "豎彎鉤 (豎90+弧r25+橫+鉤)", IOS_CANVAS, sw, 4, [("h", 350, (285, 315)), ("v", 355, (410, 440))])
    add("I08_pie", "撇 斜 100x100", IOS_CANVAS, [(450, 300), (350, 400)], 4, [("v", 400, (330, 380))])
    add("I09_dot", "點 約 10x12", IOS_CANVAS, [(400, 300), (406, 310)], 4, [])
    add("I10_edge_corner", "橫折貼左上角邊界 (min_x=min_y=0)", IOS_CANVAS, [(2, 2), (80, 2), (80, 70)], 4,
        [("v", 40, (0, 15)), ("h", 50, (65, 95))])
    add("W01_web_vertical", "Web demo 豎 8x200 (256 畫布)", WEB_CANVAS, [(128, 28), (128, 228)], 8, [("h", 128, (115, 142))])
    add("W02_web_hengzhe", "Web demo 橫折 (256 畫布)", WEB_CANVAS, [(50, 60), (200, 60), (200, 200)], 8,
        [("v", 125, (45, 75)), ("h", 130, (185, 215))])
    return S


# --------------------------------------------------------------------------
# 指標
# --------------------------------------------------------------------------


def ink(a, thr=128):
    return np.asarray(a) < thr


def iou_xor(a, b, thr=128):
    A, B = ink(a, thr), ink(b, thr)
    union = int((A | B).sum())
    inter = int((A & B).sum())
    na = int(A.sum())
    return dict(iou=inter / union if union else float("nan"),
                xor_over_ink=int((A ^ B).sum()) / na if na else float("nan"),
                area_ratio=int(B.sum()) / na if na else float("nan"),
                ink_px_orig=na, ink_px_rec=int(B.sum()))


def coverage(a):
    return (255.0 - np.asarray(a, dtype=np.float64)) / 255.0


def centroid(a):
    c = coverage(a)
    s = c.sum()
    ys, xs = np.mgrid[0:c.shape[0], 0:c.shape[1]]
    return ((xs + 0.5) * c).sum() / s, ((ys + 0.5) * c).sum() / s


def probe_width(a, probe):
    kind, pos, (lo, hi) = probe
    arr = np.asarray(a)
    if kind == "v":   # 垂直線 x=pos，沿 y 取 [lo,hi)
        line = arr[lo:hi, int(pos)]
    else:             # 水平線 y=pos，沿 x 取 [lo,hi)
        line = arr[int(pos), lo:hi]
    eqw = float((255.0 - line.astype(np.float64)).sum() / 255.0)
    run = int((line < 128).sum())
    return eqw, run


def diff_overlay(orig, rec):
    A, B = ink(orig), ink(rec)
    h, w = A.shape
    out = np.full((h, w, 3), 255, np.uint8)
    out[A & B] = (0, 0, 0)
    out[A & ~B] = (220, 30, 30)    # 只在原圖有 = 紅
    out[~A & B] = (30, 60, 220)    # 只在還原有 = 藍
    return Image.fromarray(out, "RGB")


def zoom_crop(img, box, target=480, resample=Image.NEAREST):
    c = img.crop(box)
    s = max(1, min(target // max(c.size), 40))
    return c.resize((c.size[0] * s, c.size[1] * s), resample), s


def checker(size, cell=8):
    w, h = size
    a = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // cell + xx // cell) % 2).astype(bool)
    a[m] = (235, 235, 235)
    a[~m] = (200, 200, 200)
    return Image.fromarray(a, "RGB")


def rgba_on_checker(rgba):
    base = checker(rgba.size).convert("RGBA")
    base.alpha_composite(rgba)
    return base.convert("RGB")


def label_font():
    try:
        return ImageFont.load_default(size=13)
    except TypeError:
        return ImageFont.load_default()


def montage(panels, path, panel=240):
    font = label_font()
    pad, lab = 8, 34
    n = len(panels)
    W = n * (panel + pad) + pad
    H = panel + lab + 2 * pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(panels):
        x = pad + i * (panel + pad)
        im = im.convert("RGB")
        s = min(panel / im.size[0], panel / im.size[1])
        rs = Image.NEAREST
        im2 = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), rs)
        sheet.paste(im2, (x + (panel - im2.size[0]) // 2, pad + lab + (panel - im2.size[1]) // 2))
        d.rectangle([x, pad + lab, x + panel, pad + lab + panel], outline=(180, 180, 180))
        d.text((x + 2, pad), title, fill=(0, 0, 0), font=font)
        d.text((x + 2, pad + 16), f"{im.size[0]}x{im.size[1]} px", fill=(90, 90, 90), font=font)
    sheet.save(path)


def best_shift_iou(orig, rec, rng=3, thr=128):
    """把還原圖平移 [-rng,rng] 的整數像素，找最大 IoU；用來把「整體位移」與「形狀變化」分開。"""
    A = ink(orig, thr)
    B0 = ink(rec, thr)
    best = (-1.0, (0, 0))
    h, w = A.shape
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            B = np.zeros_like(B0)
            ys0, ys1 = max(0, dy), min(h, h + dy)
            xs0, xs1 = max(0, dx), min(w, w + dx)
            B[ys0:ys1, xs0:xs1] = B0[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            u = int((A | B).sum())
            v = int((A & B).sum()) / u if u else float("nan")
            if v > best[0]:
                best = (v, (dx, dy))
    return best  # (iou, (dx, dy)) ：dx,dy 是「把還原圖往右/下平移多少 px 會最貼合原圖」


def mid_gray_px(a, lo=32, hi=224):
    arr = np.asarray(a)
    return int(((arr > lo) & (arr < hi)).sum())


PHASES = [(i * 0.25, j * 0.25) for i in range(4) for j in range(4)]


def phase_sweep(s, filt=Image.BILINEAR):
    """把整條筆跡平移 0/.25/.5/.75 px（x,y 各 4 種，共 16 種相位）後重跑，
    看結果對「筆跡落在像素格上的相位」有多敏感。"""
    out = dict(final_iou=[], r3_iou=[], best_iou=[], area_ratio=[], w_ratio=[], crop_out=0)
    for dx, dy in PHASES:
        pts = [(x + dx, y + dy) for x, y in s["pts"]]
        img = render_stroke(s["size"], pts, s["width"])
        P = run_pipeline(img, filt)
        sq = P["square"]
        _, _, _, _, _, R3 = stage_recon_square(sq, P["side"], filt)
        out["r3_iou"].append(iou_xor(sq, R3)["iou"])
        m = iou_xor(img, P["recon_full"])
        out["final_iou"].append(m["iou"])
        out["area_ratio"].append(m["area_ratio"])
        out["best_iou"].append(best_shift_iou(img, P["recon_full"])[0])
        if s["probes"]:
            eo, _ = probe_width(img, s["probes"][0])
            er, _ = probe_width(P["recon_full"], s["probes"][0])
            out["w_ratio"].append(er / eo if eo else float("nan"))
        if P["meta"]["min_x"] - 1 < 0 or P["meta"]["min_y"] - 1 < 0:
            out["crop_out"] += 1
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def stage_recon_square(sq, side, filt):
    """在「正方形座標系」內重建 R1..R3，用來定位誤差出現在哪一步。"""
    inp = sq.resize((CANVAS_SIZE, CANVAS_SIZE))
    R1 = inp.resize((side, side), filt)                                   # 只有縮放往返
    rt = tensor_roundtrip(inp)
    R2 = rt.resize((side, side), filt)                                    # + tensor 8-bit 量化
    tinted = _tint_strokes(rt, (0, 0, 0, 255))
    R3 = over_white(tinted.resize((side, side), filt))                    # + 去背上色(門檻)後再縮放
    return inp, rt, tinted, R1, R2, R3


def main():
    api_root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "smart-calligraphy-api"
    inference_py = api_root / "app" / "inference.py"
    verbatim = verify_verbatim(inference_py)
    print("AST 與 inference.py 一致:", verbatim)
    if not all(verbatim.values()):
        print("警告：複製的函式與原始碼不一致，結果不可信", file=sys.stderr)

    STAGES.mkdir(parents=True, exist_ok=True)
    results = []
    for s in build_samples():
        img = render_stroke(s["size"], s["pts"], s["width"])
        sdir = STAGES / s["id"]
        sdir.mkdir(exist_ok=True)

        rec = {"id": s["id"], "desc": s["desc"], "canvas": f"{s['size'][0]}x{s['size'][1]}", "line_width": s["width"]}
        runs = {}
        for fname, filt in (("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC)):
            runs[fname] = run_pipeline(img, filt)

        P = runs["bilinear"]
        meta = P["meta"]
        side = P["side"]
        rec.update(
            bbox_w=meta["width"], bbox_h=meta["height"], min_x=meta["min_x"], min_y=meta["min_y"], side=side,
            scale_up=CANVAS_SIZE / side, x_pos=P["x_pos"], y_pos=P["y_pos"],
        )
        # 真實墨跡 bbox（像素邊緣座標）中心 vs 回傳的 x/y_position
        A = np.asarray(img) < 100
        ys, xs = np.where(A)
        true_cx = (xs.min() + xs.max() + 1) / 2
        true_cy = (ys.min() + ys.max() + 1) / 2
        rec.update(true_cx=true_cx, true_cy=true_cy, dx_pos=P["x_pos"] - true_cx, dy_pos=P["y_pos"] - true_cy)

        # 正方形座標系的逐步誤差（bilinear / bicubic）
        sq = P["square"]
        for fname, filt in (("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC)):
            inp, rt, tinted, R1, R2, R3 = stage_recon_square(sq, side, filt)
            rec[f"R1_iou_{fname}"] = iou_xor(sq, R1)["iou"]
            rec[f"R2_iou_{fname}"] = iou_xor(sq, R2)["iou"]
            m3 = iou_xor(sq, R3)
            rec[f"R3_iou_{fname}"] = m3["iou"]
            rec[f"R1_mad_{fname}"] = float(np.abs(np.asarray(sq, np.float64) - np.asarray(R1, np.float64)).mean())
            if fname == "bilinear":
                saved = dict(inp=inp, rt=rt, tinted=tinted, R1=R1, R2=R2, R3=R3)

        # 全畫布最終結果
        for fname in ("bilinear", "bicubic"):
            R = runs[fname]
            m = iou_xor(img, R["recon_full"])
            rec[f"final_iou_{fname}"] = m["iou"]
            rec[f"final_xor_over_ink_{fname}"] = m["xor_over_ink"]
            rec[f"final_area_ratio_{fname}"] = m["area_ratio"]
            cov_o = coverage(img).sum()
            cov_r = coverage(R["recon_full"]).sum()
            rec[f"final_cov_ratio_{fname}"] = cov_r / cov_o
            ox, oy = centroid(img)
            rx, ry = centroid(R["recon_full"])
            rec[f"centroid_dx_{fname}"] = rx - ox
            rec[f"centroid_dy_{fname}"] = ry - oy
            rec[f"final_iou_thr200_{fname}"] = iou_xor(img, R["recon_full"], 200)["iou"]
        m = iou_xor(img, P["recon_full"])
        rec.update(ink_px_orig=m["ink_px_orig"], ink_px_rec=m["ink_px_rec"])
        bi, bs = best_shift_iou(img, P["recon_full"])
        rec.update(best_shift_iou=bi, best_shift=list(bs))
        rec.update(mid_gray_orig=mid_gray_px(sq), mid_gray_R1=mid_gray_px(saved["R1"]),
                   mid_gray_recon_full=mid_gray_px(P["recon_full"]), mid_gray_orig_full=mid_gray_px(img))
        rec["sweep"] = phase_sweep(s)

        # 線寬探針
        pr = []
        for probe in s["probes"]:
            ew_o, run_o = probe_width(img, probe)
            ew_r, run_r = probe_width(P["recon_full"], probe)
            pr.append(dict(probe=f"{probe[0]}@{probe[1]}", eqw_orig=ew_o, eqw_rec=ew_r, run_orig=run_o, run_rec=run_r))
        rec["probes"] = pr

        # 去背上色三類像素統計（256x256 的 tinted）
        t = np.asarray(P["tinted"])
        alpha = t[..., 3]
        rec["tint_transparent"] = int((alpha == 0).sum())
        rec["tint_color"] = int(((alpha == 255) & (t[..., 0] == 0) & (t[..., 1] == 0) & (t[..., 2] == 0)).sum())
        rec["tint_fringe_opaque_gray"] = int(((alpha == 255) & ~((t[..., 0] == 0) & (t[..., 1] == 0) & (t[..., 2] == 0))).sum())
        # 8-bit 量化造成灰階值變化的像素數
        a0 = np.asarray(P["inp"], np.int16)
        a1 = np.asarray(P["rt"], np.int16)
        rec["roundtrip_changed_px"] = int((a0 != a1).sum())
        rec["roundtrip_max_abs"] = int(np.abs(a0 - a1).max())
        # fitbest 邊界：是否因貼邊而產生 crop 超出影像（PIL crop 超出範圍補 0=黑）
        rec["crop_outside_top_left"] = bool(meta["min_x"] - 1 < 0 or meta["min_y"] - 1 < 0)
        if rec["crop_outside_top_left"]:
            sqa = np.asarray(sq)
            rec["square_black_border_px"] = int(((sqa[0, :] == 0).sum()) + ((sqa[:, 0] == 0).sum()))

        # ---- 存各階段圖 ----
        img.save(sdir / "00_input_full.png")
        P["square"].save(sdir / "01_fitbest_square.png")
        saved["inp"].save(sdir / "02_model_input_256.png")
        saved["rt"].save(sdir / "03_after_tensor_roundtrip_256.png")
        P["tinted"].save(sdir / "04_tinted_rgba_256.png")
        rgba_on_checker(P["tinted"]).save(sdir / "04b_tinted_on_checker_256.png")
        P["layer"].save(sdir / "05_layer_scaled_back_bilinear.png")
        P["recon_full"].save(sdir / "06_recon_full_bilinear.png")
        ov = diff_overlay(img, P["recon_full"])
        ov.save(sdir / "07_diff_full_bilinear.png")
        ux0 = int(max(0, min(meta["min_y"], P["origin"][0]) - 6))
        uy0 = int(max(0, min(meta["min_x"], P["origin"][1]) - 6))
        ux1 = int(min(img.size[0], max(meta["min_y"] + meta["width"], P["origin"][0] + side) + 6))
        uy1 = int(min(img.size[1], max(meta["min_x"] + meta["height"], P["origin"][1] + side) + 6))
        box = (ux0, uy0, ux1, uy1)
        z, zs = zoom_crop(ov, box)
        z.save(sdir / "08_diff_zoom_bilinear.png")
        in_view, _ = zoom_crop(img.convert("RGB"), box)
        montage([
            ("A input (crop view)", in_view),
            ("B fitbest square", P["square"]),
            ("C model input 256", saved["inp"]),
            ("D tensor roundtrip", saved["rt"]),
            ("E tinted RGBA", rgba_on_checker(P["tinted"])),
            ("F layer scaled back", rgba_on_checker(P["layer"])),
            ("G diff (red=orig only, blue=recon only)", z),
        ], sdir / "montage.png")
        rec["stage_dir"] = str(sdir.relative_to(HERE.parent)).replace("\\", "/")
        results.append(rec)
        print(f"[{s['id']}] side={side} scale={CANVAS_SIZE/side:.2f}x  IoU(bilinear) R1={rec['R1_iou_bilinear']:.3f} "
              f"R2={rec['R2_iou_bilinear']:.3f} R3={rec['R3_iou_bilinear']:.3f} final={rec['final_iou_bilinear']:.3f}")

    meta_out = dict(pillow=PIL_VERSION, numpy=np.__version__, verbatim_check=verbatim, ios_canvas=IOS_CANVAS,
                    web_canvas=WEB_CANVAS, supersample=SS)
    (HERE / "roundtrip_results.json").write_text(
        json.dumps(dict(meta=meta_out, samples=results), ensure_ascii=False, indent=2), encoding="utf-8")
    write_tables(meta_out, results)


def f(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"


def write_tables(meta_out, R):
    L = []
    L.append(f"_環境：Pillow {meta_out['pillow']}、numpy {meta_out['numpy']}；"
             f"fitbest/_tint_strokes 與 inference.py 的 AST 一致性：{meta_out['verbatim_check']}_\n")

    L.append("**表 5-1　每個樣本的幾何（fitbest 之後）**\n")
    L.append("| 樣本 | 說明 | 畫布 | bbox 寬x高 (fitbest) | side=max(w,h) | 縮放倍率 256/side | 回傳 x_pos,y_pos | 真實墨跡 bbox 中心 | 差 (dx,dy) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in R:
        L.append(f"| {r['id']} | {r['desc']} | {r['canvas']} | {r['bbox_w']}x{r['bbox_h']} | {r['side']} | "
                 f"{r['scale_up']:.2f}x | {r['x_pos']:.1f}, {r['y_pos']:.1f} | {r['true_cx']:.1f}, {r['true_cy']:.1f} | "
                 f"{r['dx_pos']:+.1f}, {r['dy_pos']:+.1f} |")

    L.append("\n**表 5-2　往返 IoU（墨跡門檻 gray<128）——逐步累積；回傳縮放濾波器以 PIL BILINEAR 近似**\n")
    L.append("| 樣本 | R1 只有 256 縮放往返 | R2 +tensor 8-bit | R3 +去背上色門檻 | 最終（全畫布，含貼回位置） | 最終 xor/原墨跡像素 | 最終墨跡面積比 (還原/原) | 最終總墨量比 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in R:
        L.append(f"| {r['id']} | {f(r['R1_iou_bilinear'])} | {f(r['R2_iou_bilinear'])} | {f(r['R3_iou_bilinear'])} | "
                 f"{f(r['final_iou_bilinear'])} | {f(r['final_xor_over_ink_bilinear'])} | "
                 f"{f(r['final_area_ratio_bilinear'])} | {f(r['final_cov_ratio_bilinear'])} |")

    L.append("\n**表 5-3　濾波器假設的敏感度（BILINEAR vs BICUBIC）與不同墨跡門檻**\n")
    L.append("| 樣本 | 最終 IoU bilinear | 最終 IoU bicubic | 最終 IoU bilinear (門檻200) | R1 灰階平均絕對差 (bilinear) | 質心偏移 dx,dy px (bilinear) |")
    L.append("|---|---|---|---|---|---|")
    for r in R:
        L.append(f"| {r['id']} | {f(r['final_iou_bilinear'])} | {f(r['final_iou_bicubic'])} | "
                 f"{f(r['final_iou_thr200_bilinear'])} | {f(r['R1_mad_bilinear'])} | "
                 f"{r['centroid_dx_bilinear']:+.2f}, {r['centroid_dy_bilinear']:+.2f} |")

    L.append("\n**表 5-4　線寬變化（沿垂直於筆劃的探針線；等效寬度 = 探針線上 (255-gray)/255 的總和，單位 px）**\n")
    L.append("| 樣本 | 探針 | 原圖等效寬度 | 還原等效寬度 | 比值 | 原圖 <128 連續像素 | 還原 <128 連續像素 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in R:
        if not r["probes"]:
            L.append(f"| {r['id']} | （無探針：筆劃太短） | - | - | - | - | - |")
        for p in r["probes"]:
            ratio = p["eqw_rec"] / p["eqw_orig"] if p["eqw_orig"] else float("nan")
            L.append(f"| {r['id']} | {p['probe']} | {p['eqw_orig']:.2f} | {p['eqw_rec']:.2f} | {f(ratio, 2)} | "
                     f"{p['run_orig']} | {p['run_rec']} |")

    L.append("\n**表 5-5　去背上色三類像素與 8-bit 量化（256x256 模型輸入/輸出；模型以恆等代替）**\n")
    L.append("| 樣本 | 全透明 (luma>240) | 純色 (luma<200) | 不透明灰邊 (200–240，原樣保留) | tensor 往返改變的像素數 | 往返最大絕對差 | fitbest 的 crop 超出左/上邊界 | 黑邊像素數 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in R:
        L.append(f"| {r['id']} | {r['tint_transparent']} | {r['tint_color']} | {r['tint_fringe_opaque_gray']} | "
                 f"{r['roundtrip_changed_px']} | {r['roundtrip_max_abs']} | {'是' if r['crop_outside_top_left'] else '否'} | "
                 f"{r.get('square_black_border_px', '-')} |")


    L.append("\n**表 5-6　把「整體位移」與「形狀變化」分開：最佳整數位移對齊後的 IoU，以及灰階邊緣像素數**\n")
    L.append("| 樣本 | 最終 IoU（未對齊） | 最佳對齊 IoU | 對齊所需位移 dx,dy (px，還原圖往右/下) | 邊緣灰階像素數 (32<g<224)：正方形原圖 → 256 往返後(R1) | 全畫布：原圖 → 還原圖 |")
    L.append("|---|---|---|---|---|---|")
    for r in R:
        L.append(f"| {r['id']} | {f(r['final_iou_bilinear'])} | {f(r['best_shift_iou'])} | "
                 f"{r['best_shift'][0]:+d}, {r['best_shift'][1]:+d} | {r['mid_gray_orig']} → {r['mid_gray_R1']} | "
                 f"{r['mid_gray_orig_full']} → {r['mid_gray_recon_full']} |")

    L.append("\n**表 5-7　次像素相位敏感度（整條筆跡平移 0/.25/.5/.75 px，x、y 各 4 種，共 16 種相位；BILINEAR）**\n")
    L.append("| 樣本 | R3 IoU min/中位/max | 最終 IoU min/中位/max | 最佳對齊 IoU min/中位/max | 墨跡面積比 min/中位/max | 第一探針線寬比 min/中位/max | crop 超出左/上界的相位數 |")
    L.append("|---|---|---|---|---|---|---|")
    def tri(v, n=3):
        v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
        if not v:
            return "-"
        v = sorted(v)
        return f"{v[0]:.{n}f} / {v[len(v)//2]:.{n}f} / {v[-1]:.{n}f}"
    for r in R:
        sw = r["sweep"]
        L.append(f"| {r['id']} | {tri(sw['r3_iou'])} | {tri(sw['final_iou'])} | {tri(sw['best_iou'])} | "
                 f"{tri(sw['area_ratio'])} | {tri(sw['w_ratio'], 2)} | {sw['crop_out']}/16 |")

    (HERE / "roundtrip_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
