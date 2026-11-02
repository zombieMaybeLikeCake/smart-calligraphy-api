校準：zi2zi G = 17.84 GMACs/forward，README p95 35.4 ms ÷ 2 次 forward → 0.99 ms/GMAC（上界，含 HTTP/PNG 開銷）

| net | params_M | GMACs | steps | est_ms_p95 | vram_train_b8_GB | vram_train_b16_GB | note |
|---|---|---|---|---|---|---|---|
| zi2zi UNetGenerator 256 | 55.5 | 17.84 | 2 | 35.4 | 1.75 | 2.01 | 現況：forward() 跑 netG 兩次（model.py:100-101） |
| zi2zi UNetGenerator 256 | 55.5 | 17.84 | 1 | 17.7 | 1.75 | 2.01 | 低成本修正：推論只跑一次 netG（直接呼叫 netG 而非 model.forward） |
| diffusion UNet 64px base64 mults(1, 2, 4) | 11.7 | 6.75 | 1 | 6.7 | 1.08 | 1.38 | 擴散/流匹配 64px，1 步 |
| diffusion UNet 64px base64 mults(1, 2, 4) | 11.7 | 6.75 | 4 | 26.8 | 1.08 | 1.38 | 擴散/流匹配 64px，4 步 |
| diffusion UNet 64px base64 mults(1, 2, 4) | 11.7 | 6.75 | 8 | 53.6 | 1.08 | 1.38 | 擴散/流匹配 64px，8 步 |
| diffusion UNet 64px base64 mults(1, 2, 4) | 11.7 | 6.75 | 20 | 134.0 | 1.08 | 1.38 | 擴散/流匹配 64px，20 步 |
| diffusion UNet 128px base64 mults(1, 2, 4) | 11.7 | 27.41 | 1 | 27.2 | 2.03 | 3.27 | 擴散/流匹配 128px，1 步 |
| diffusion UNet 128px base64 mults(1, 2, 4) | 11.7 | 27.41 | 4 | 108.8 | 2.03 | 3.27 | 擴散/流匹配 128px，4 步 |
| diffusion UNet 128px base64 mults(1, 2, 4) | 11.7 | 27.41 | 8 | 217.5 | 2.03 | 3.27 | 擴散/流匹配 128px，8 步 |
| diffusion UNet 128px base64 mults(1, 2, 4) | 11.7 | 27.41 | 20 | 543.8 | 2.03 | 3.27 | 擴散/流匹配 128px，20 步 |
| diffusion UNet 256px base64 mults(1, 2, 4, 4) | 20.7 | 114.47 | 4 | 454.2 | 5.86 | 10.8 | 擴散 256px（為了對齊現況解析度），4 步 |

軌跡 Transformer（L=128, d=256, 6 層）：0.654 GMACs → 估 0.6 ms；若輸出仍是 256 點陣再接一個小解碼器（≈ 現況 G 的一半）合計 < 9 ms
