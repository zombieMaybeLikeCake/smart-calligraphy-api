"""診斷腳本 4：量化你用網頁 demo 打 API 得到的真實輸出（yong_eight_strokes_samples.zip）。
用法：python analysis/diag_real_samples.py <解壓後的樣本資料夾>
量：模型空間的輸入線寬、輸出最粗線寬、墨量比、輸出 bbox 佔比、是否觸到畫布邊（被裁切）、
質心漂移、recall（輸入墨跡被輸出±12px 覆蓋的比例）、precision（輸出落在輸入±12px 內的比例）。
只需 numpy + Pillow。"""
import json, sys, numpy as np
from PIL import Image
root = sys.argv[1]
def fitbest256(img):
    g = np.asarray(img.convert("L")); rows, cols = np.where(g < 100)
    mn_x, mx_x, mn_y, mx_y = rows.min(), rows.max(), cols.min(), cols.max()
    cr = Image.fromarray(g).crop((mn_y-1, mn_x-1, mx_y+1, mx_x+1)); w, h = mx_y-mn_y+2, mx_x-mn_x+2; S = max(w, h)
    cv = Image.new("L", (S, S), 255); cv.paste(cr, (0, (S-h)//2) if w > h else ((S-w)//2, 0))
    return np.asarray(cv.resize((256, 256))) < 100, S
def erode_depth(b):
    d = 0; b = b.copy()
    while b.any():
        p = np.pad(b, 1); e = p[1:-1,1:-1].copy()
        for dy in (-1,0,1):
            for dx in (-1,0,1): e &= p[1+dy:p.shape[0]-1+dy, 1+dx:p.shape[1]-1+dx]
        b = e; d += 1
    return 2*d-1
def dilate(b, r):
    for _ in range(r):
        p = np.pad(b, 1); e = p[1:-1,1:-1].copy()
        for dy in (-1,0,1):
            for dx in (-1,0,1): e |= p[1+dy:p.shape[0]-1+dy, 1+dx:p.shape[1]-1+dx]
        b = e
    return b
samples = json.load(open(f"{root}/samples.json"))
print("| run | id | 筆畫 | 放大倍 | 輸入線寬@256 | 輸出最粗線寬 | 輸出墨量/輸入墨量 | 輸出bbox邊長/256 | 觸邊 | 中心漂移(px) | recall(輸入被覆蓋) | precision(輸出落在輸入±12px) |")
print("|---|" * 12)
rows=[]
for s in samples:
    run = s["output_file"].split("/")[1]
    inp = Image.open(f"{root}/{s['input_file']}").convert("RGBA"); bg = Image.new("RGBA", inp.size, (255,255,255,255)); bg.alpha_composite(inp)
    im, S = fitbest256(bg.convert("L"))
    out = Image.open(f"{root}/{s['output_file']}").convert("RGBA"); a = np.asarray(out)
    om = a[...,3] > 128  # 不透明 = 墨
    r, c = np.where(om); bb = (c.max()-c.min()+1, r.max()-r.min()+1) if r.size else (0,0)
    edge = r.size and (r.min()==0 or c.min()==0 or r.max()==255 or c.max()==255)
    ri, ci = np.where(im); cin = (ci.mean(), ri.mean()); cout = (c.mean(), r.mean()) if r.size else (np.nan,np.nan)
    drift = float(np.hypot(cout[0]-cin[0], cout[1]-cin[1]))
    tol = dilate(im, 12)
    recall = float((dilate(om,12) & im).sum()/im.sum()); prec = float((om & tol).sum()/max(1,om.sum()))
    row = dict(run=run, id=s["id"], name=s["name_zh"], up=s["upscale_factor_before_model"], w_in=erode_depth(im), w_out=erode_depth(om), ink=round(om.sum()/im.sum(),1), bbox=round(max(bb)/256,2), edge="是" if edge else "否", drift=round(drift,1), recall=round(recall,2), prec=round(prec,2))
    rows.append(row); print("| " + " | ".join(str(v) for v in row.values()) + " |")
json.dump(rows, open("quant_yong.json","w"), ensure_ascii=False, indent=1)
for run in sorted(set(r["run"] for r in rows)):
    rs=[r for r in rows if r["run"]==run]
    print(f"\n[{run}] 輸出線寬 {min(r['w_out'] for r in rs)}–{max(r['w_out'] for r in rs)} px；墨量比 {min(r['ink'] for r in rs)}–{max(r['ink'] for r in rs)}×；觸邊 {sum(r['edge']=='是' for r in rs)}/{len(rs)}；recall 中位 {np.median([r['recall'] for r in rs]):.2f}；precision 中位 {np.median([r['prec'] for r in rs]):.2f}")
    hi=[r for r in rs if r['up']>=5]; lo=[r for r in rs if r['up']<3]
    if hi and lo: print(f"   放大≥5× 的 recall 中位 {np.median([r['recall'] for r in hi]):.2f} / precision {np.median([r['prec'] for r in hi]):.2f}；放大<3× 的 recall {np.median([r['recall'] for r in lo]):.2f} / precision {np.median([r['prec'] for r in lo]):.2f}")
