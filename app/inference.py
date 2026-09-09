"""模型載入與推論邏輯。純函式，不含任何 FastAPI / web 相關程式碼。

這裡取代了原本 strokeStytleChangeServerV2.py 的 do_POST 邏輯，但做了一個
關鍵修正：原本程式呼叫的 `model.singalsample(batch, infer_dir)` 在目前
repo 的 model/model.py 裡並不存在（只有 sample()，是另一個東西），呼叫下去
會直接 AttributeError。這裡改成直接用 model.set_input() + model.forward()
跑推論、在記憶體裡把輸出 tensor 轉成圖片，不寫檔案到磁碟——這樣服務才是
無狀態的，也才符合「純函式」的要求。

裝載的模型用 ModelHandle 包起來當回傳值，不用模組層級的全域變數存單一個
模型實例——階段二要同時常駐 word/stroke 兩個模型、各自綁一個 FastAPI 端點，
全域單例裝不下兩個。
"""
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import Zi2ZiModel

CANVAS_SIZE = 256


@dataclass
class ModelHandle:
    """load_model() 的回傳值，把模型實例跟它實際跑在哪個裝置綁在一起。
    predict() 認的是這個 handle，不是任何全域狀態。"""

    model: Zi2ZiModel
    device: str
    version: str


def load_model(
    model_dir: str,
    epoch: int,
    device: str = "cuda",
    embedding_num: int = 40,
    embedding_dim: int = 128,
    version: str = "",
) -> ModelHandle:
    """服務啟動時對每個模型呼叫一次。model_dir 底下要有 f"{epoch}_net_G.pth" 和
    f"{epoch}_net_D.pth"（Zi2ziModel.load_networks 的命名規則）。"""
    if device == "cuda" and not torch.cuda.is_available():
        print(f"[WARN] DEVICE=cuda 但找不到可用 GPU，{model_dir} 自動降級為 CPU")
        device = "cpu"

    gpu_ids = ["cuda:0"] if device == "cuda" else None

    model = Zi2ZiModel(
        input_nc=1,
        embedding_num=embedding_num,
        embedding_dim=embedding_dim,
        Lconst_penalty=15,
        Lcategory_penalty=1.0,
        save_dir=model_dir,
        gpu_ids=gpu_ids,
        is_training=False,
    )
    model.setup()
    model.load_networks(epoch)

    return ModelHandle(model=model, device=device, version=version)


def fitbest(image: Image.Image) -> Tuple[Image.Image, dict]:
    """從原圖裡找出筆劃的邊界框，裁切後置中貼進一個正方形畫布。

    回傳 (正方形畫布圖, 座標中繼資料)。中繼資料是把生成結果對應回原圖
    座標系（xPosition/yPosition）時需要的資訊，來自原本的 fitbest()，
    這裡用 numpy 向量化取代逐像素 for 迴圈（原本 256x256 圖要跑 65536 次
    Python 迴圈，是可量測到的延遲來源）。

    命名沿用原始程式碼的慣例：min_x/max_x 對應「列」（圖片直向），
    min_y/max_y 對應「欄」（圖片橫向）——雖然跟一般 x=橫向 的直覺相反，
    但跟原本 singalereduction() 的參數順序是一致的，沿用可以避免對錯行列。
    """
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
    """白色背景去背成透明、深色筆劃塗成指定顏色，中間的反鋸齒邊緣像素原樣保留。

    向量化版本的 singalereduction()（拿掉了原本函式裡兩個沒有實際作用的
    參數 oringalimgheight/oringalimgwidth，以及算完全沒被用到的區域邊界，
    這兩塊是死碼——它是最終沒用來定位、也沒有回寫的計算，這裡略過）。
    """
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


def _image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def preprocess(image: Image.Image) -> Tuple[torch.Tensor, dict]:
    """圖片 → 模型輸入 tensor，同時回傳定位用的中繼資料。"""
    canvas, meta = fitbest(image)
    canvas = canvas.resize((CANVAS_SIZE, CANVAS_SIZE))
    tensor = transforms.Normalize(0.5, 0.5)(transforms.ToTensor()(canvas)).unsqueeze(0)
    return tensor, meta


def predict(
    handle: ModelHandle,
    image: Image.Image,
    label: int = 0,
    stroke_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
) -> dict:
    """單張圖片推論，回傳可直接 JSON 序列化的 dict。

    注意：這個模型是風格轉換（zi2zi），輸出不是分類標籤，而是一張重新
    上色/去背過的筆劃圖片＋在原圖座標系裡的位置資訊，對應 iOS App 端
    需要的 stroke overlay 格式。
    """
    tensor, meta = preprocess(image)
    label_tensor = torch.tensor([label])

    with torch.no_grad():
        handle.model.set_input(label_tensor, tensor, tensor)
        handle.model.forward()
        fake_b = handle.model.fake_B[0].detach().cpu()

    # UNetGenerator 最後一層是 Tanh，輸出範圍是 [-1, 1]，
    # 要先還原成 [0, 1] 才能轉成正常的圖片像素值。
    fake_b = (fake_b * 0.5 + 0.5).clamp(0, 1)
    generated = transforms.ToPILImage()(fake_b)
    tinted = _tint_strokes(generated, stroke_color)

    return {
        "x_position": meta["min_y"] + meta["width"] / 2,
        "y_position": meta["min_x"] + meta["height"] / 2,
        "width": max(meta["width"], meta["height"]),
        "height": max(meta["width"], meta["height"]),
        "image_base64": _image_to_base64(tinted),
    }
