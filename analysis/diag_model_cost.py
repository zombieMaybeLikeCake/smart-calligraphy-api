"""診斷腳本 3：用程式算（不心算）各方案的參數量、乘加次數（MACs）、VRAM 與延遲估算。

延遲估算方法：
  1. 以現況 zi2zi UNetGenerator（model/generators.py，ngf=64、8 層下採樣、256x256、
     embedding_dim=128）解析式算出 MACs。
  2. README 量到 /predict/stroke 穩態 p95 = 35.4 ms（含 HTTP、PNG 編解碼、fitbest、
     兩次 netG forward——model/model.py:100-101 forward() 跑了 netG 兩次：
     一次生成、一次把 fake_B 再編碼算 const loss，推論時第二次是浪費）。
     因此「每 GMAC 的毫秒數」= 35.4 / (2 * G_MACs) 是上界（把全部時間都算給 GPU），
     用它線性外推其他網路是保守（偏高）的估計。
  3. 擴散模型：取小型 UNet（通道 64/128/256，解析度 64 或 128，attention 只放最低層）
     的 MACs × 取樣步數。
VRAM 估算：參數 + Adam 狀態（fp32：4+4+8 bytes/param）+ 啟動值（以每層輸出張量
 batch*C*H*W*4 bytes 的加總 × 2 作 backward 緩衝的粗估）。
只需標準函式庫。用法：python analysis/diag_model_cost.py
"""
import json
from pathlib import Path

MEASURED_P95_MS = 35.4   # README.md「推論延遲」表
IMG = 256


def conv_macs(cin, cout, k, hout, wout, groups=1):
    return cin * cout * k * k * hout * wout // groups


def zi2zi_generator(ngf=64, num_downs=8, img=IMG, in_nc=1, emb=128):
    """逐層對照 model/generators.py 的 UnetSkipConnectionBlock 結構。"""
    macs, params, acts = 0, 0, 0
    # 下採樣通道序列（從外到內）：in->ngf->2ngf->4ngf->8ngf->8ngf*4 (num_downs-5 層)->innermost 8ngf
    chans = [in_nc, ngf, ngf * 2, ngf * 4, ngf * 8] + [ngf * 8] * (num_downs - 4)
    h = img
    downs = []
    for i in range(num_downs):
        cin, cout = chans[i], chans[i + 1]
        h //= 2
        macs += conv_macs(cin, cout, 4, h, h); params += cin * cout * 16; acts += cout * h * h
        downs.append((cin, cout, h))
    # 上採樣：innermost 輸入 8ngf+emb；其餘輸入 2*inner_nc（skip concat）
    for i in reversed(range(num_downs)):
        cin_d, cout_d, h = downs[i]
        up_in = cout_d + emb if i == num_downs - 1 else cout_d * 2
        up_out = cin_d if i != 0 else in_nc
        macs += conv_macs(up_in, up_out, 4, h * 2, h * 2); params += up_in * up_out * 16; acts += up_out * h * h * 4
    params += 40 * emb  # embedding table
    return dict(name="zi2zi UNetGenerator 256", macs=macs, params=params, acts=acts)


def small_diffusion_unet(res=64, base=64, mults=(1, 2, 4), in_ch=2, blocks=2, attn_at_lowest=True):
    """DDPM 風格小 UNet：每層 2 個 ResBlock（各 2 個 3x3 conv），下/上採樣各一，最低層 self-attention。"""
    macs, params, acts = 0, 0, 0
    h, c = res, base
    chans = [base * m for m in mults]
    macs += conv_macs(in_ch, base, 3, h, h); params += in_ch * base * 9
    skip = []
    for i, cout in enumerate(chans):
        for _ in range(blocks):
            macs += conv_macs(c, cout, 3, h, h) + conv_macs(cout, cout, 3, h, h)
            params += (c * cout + cout * cout) * 9 + 512 * cout  # time-emb linear
            acts += 2 * cout * h * h; c = cout; skip.append((c, h))
        if i < len(chans) - 1:
            h //= 2; macs += conv_macs(c, c, 3, h, h); params += c * c * 9
    if attn_at_lowest:
        n = h * h; macs += 4 * c * c * n + 2 * n * n * c; params += 4 * c * c; acts += n * n
    for i, cout in reversed(list(enumerate(chans))):
        for _ in range(blocks + 1):
            sc, _ = skip.pop() if skip else (0, 0)
            macs += conv_macs(c + sc, cout, 3, h, h) + conv_macs(cout, cout, 3, h, h)
            params += ((c + sc) * cout + cout * cout) * 9 + 512 * cout
            acts += 2 * cout * h * h; c = cout
        if i > 0:
            h *= 2; macs += conv_macs(c, c, 3, h, h); params += c * c * 9
    macs += conv_macs(c, 1, 3, h, h)
    return dict(name=f"diffusion UNet {res}px base{base} mults{mults}", macs=macs, params=params, acts=acts)


def vram_train_gb(net, batch, bytes_per_act=4, optimizer_bytes=16):
    params_gb = net["params"] * optimizer_bytes / 1e9       # fp32 權重 4 + grad 4 + Adam m,v 8
    acts_gb = net["acts"] * batch * bytes_per_act * 2 / 1e9  # forward 保留 + backward 緩衝 ×2 粗估
    return params_gb + acts_gb + 0.6                         # +0.6 GB CUDA context/cudnn workspace


def main():
    G = zi2zi_generator()
    ms_per_gmac = MEASURED_P95_MS / (2 * G["macs"] / 1e9)   # forward() 跑兩次 netG
    rows = []
    def add(net, steps=1, note=""):
        lat = net["macs"] / 1e9 * ms_per_gmac * steps
        rows.append(dict(net=net["name"], params_M=round(net["params"] / 1e6, 1), GMACs=round(net["macs"] / 1e9, 2),
                         steps=steps, est_ms_p95=round(lat, 1),
                         vram_train_b8_GB=round(vram_train_gb(net, 8), 2), vram_train_b16_GB=round(vram_train_gb(net, 16), 2), note=note))
    add(G, 2, "現況：forward() 跑 netG 兩次（model.py:100-101）")
    add(G, 1, "低成本修正：推論只跑一次 netG（直接呼叫 netG 而非 model.forward）")
    for res in (64, 128):
        for steps in (1, 4, 8, 20):
            add(small_diffusion_unet(res), steps, f"擴散/流匹配 {res}px，{steps} 步")
    add(small_diffusion_unet(256, base=64, mults=(1, 2, 4, 4)), 4, "擴散 256px（為了對齊現況解析度），4 步")
    hdr = ["net", "params_M", "GMACs", "steps", "est_ms_p95", "vram_train_b8_GB", "vram_train_b16_GB", "note"]
    print(f"校準：zi2zi G = {G['macs']/1e9:.2f} GMACs/forward，README p95 {MEASURED_P95_MS} ms ÷ 2 次 forward → {ms_per_gmac:.2f} ms/GMAC（上界，含 HTTP/PNG 開銷）\n")
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    # 軌跡模型（方向 B 輸入軌跡）：Transformer encoder 小模型，序列長度 L=128 點、d=256、6 層
    L, d, layers = 128, 256, 6
    tr_macs = layers * (4 * L * d * d + 2 * L * L * d + 8 * L * d * d)
    print(f"\n軌跡 Transformer（L={L}, d={d}, {layers} 層）：{tr_macs/1e9:.3f} GMACs → 估 {tr_macs/1e9*ms_per_gmac:.1f} ms；"
          f"若輸出仍是 256 點陣再接一個小解碼器（≈ 現況 G 的一半）合計 < {G['macs']/2e9*ms_per_gmac + tr_macs/1e9*ms_per_gmac:.0f} ms")
    Path(__file__).with_suffix(".json").write_text(json.dumps(dict(ms_per_gmac=ms_per_gmac, rows=rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
