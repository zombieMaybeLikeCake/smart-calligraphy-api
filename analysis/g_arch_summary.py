#!/usr/bin/env python
"""依 smart-calligraphy-api/model/generators.py 的建構邏輯，用純 Python 列出 UNetGenerator 每層
shape 與參數量（不需要 torch）。這是「從原始碼推算」，不是載入權重後的實測；
D 的對應實測見 analysis/checkpoint_D_inspect.txt。"""
NGF, EMB, NDOWNS, IN_NC, OUT_NC, NUM_EMB = 64, 128, 8, 1, 1, 40
# (名稱, outer_nc, inner_nc, input_nc, 種類)
blocks = [("B0 outermost", OUT_NC, NGF, IN_NC, "outer"),
          ("B1", NGF, NGF * 2, None, "mid"), ("B2", NGF * 2, NGF * 4, None, "mid"), ("B3", NGF * 4, NGF * 8, None, "mid")]
for i in range(NDOWNS - 5):
    blocks.append((f"B{4 + i} (ngf*8)", NGF * 8, NGF * 8, None, "mid"))
blocks.append((f"B{4 + NDOWNS - 5} innermost", NGF * 8, NGF * 8, None, "inner"))
res, total = 256, 0
print(f"{'block':20s} {'down conv (in->out, k4 s2 p1)':34s} {'feature map after down':26s} {'up convT':34s} {'params':>10s}")
for name, outer, inner, inp, kind in blocks:
    inp = inp or outer
    down = inp * inner * 16                       # bias=False (BatchNorm 時 use_bias=False)
    res //= 2
    if kind == "outer":
        up_in = inner * 2; up = up_in * outer * 16 + outer          # outermost upconv 有 bias
        bn = 0
    elif kind == "inner":
        up_in = inner + EMB; up = up_in * outer * 16
        bn = 2 * outer                                              # 只有 up 端 BN
    else:
        up_in = inner * 2; up = up_in * outer * 16
        bn = 2 * inner + 2 * outer                                  # down BN + up BN
    p = down + up + bn
    total += p
    print(f"{name:20s} {f'{inp}->{inner}':34s} {f'{inner}x{res}x{res}':26s} {f'{up_in}->{outer}':34s} {p:>10,d}")
emb = NUM_EMB * EMB
total += emb
print(f"{'embedder':20s} nn.Embedding({NUM_EMB},{EMB}) = {emb:,d}")
print(f"G 可訓練參數總計（含 BN 的 weight/bias，不含 BN running stats）: {total:,d}  ≈ {total*4/1e6:.1f} MB (float32)")
print("bottleneck: 512x1x1 與 style(128) 在 channel 維度 concat -> 640 channel 再進 ConvTranspose2d")
