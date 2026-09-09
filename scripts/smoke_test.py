"""階段一驗收用的手動煙霧測試：不透過掃描圖片，而是用字型畫一個字當作
「手寫輸入」（跟原本 GET /goalword 端點合成測試圖的邏輯一樣），跑過
fitbest -> 模型推論 -> 上色去背，確認整條 pipeline 真的能跑、checkpoint
真的能載入，而不是只驗證 import 不報錯。

用法：
    python scripts/smoke_test.py
"""
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from app import inference

ROOT = Path(__file__).resolve().parent.parent


def draw_single_char(ch: str, font: ImageFont.FreeTypeFont, canvas_size: int) -> Image.Image:
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (0, 0, 0), font=font)
    return img.convert("L")


def main() -> None:
    print(f"torch device 期望: cuda（GPU 不可用會自動降級，見下方 warning）")
    t0 = time.perf_counter()
    inference.load_model(
        model_dir=str(ROOT / "models" / "checkpoint_word"),
        epoch=6000,
        device="cuda",
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[OK] load_model 完成，實際裝置: {inference.active_device}，耗時 {load_ms:.1f} ms")

    font = ImageFont.truetype(str(ROOT / "fonts" / "BiauKai.ttf"), size=200)
    test_image = draw_single_char("永", font, 256)

    t0 = time.perf_counter()
    result = inference.predict(test_image, label=0)
    predict_ms = (time.perf_counter() - t0) * 1000
    print(f"[OK] predict 完成，耗時 {predict_ms:.1f} ms")
    print(f"     x_position={result['x_position']:.1f} y_position={result['y_position']:.1f} "
          f"width={result['width']} height={result['height']}")

    out_path = ROOT / "scripts" / "smoke_test_output.png"
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(result["image_base64"]))
    print(f"[OK] 生成結果存到 {out_path}")


if __name__ == "__main__":
    main()
