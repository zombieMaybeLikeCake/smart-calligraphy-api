from typing import Dict

from pydantic import BaseModel


class StrokeResult(BaseModel):
    """單張圖片推論結果。這個模型是風格轉換（zi2zi），不是分類器，所以
    輸出不是 label/confidence，而是一張重新上色/去背過的筆劃圖片，加上它
    在原圖座標系裡的位置——對應 iOS App 端拿來疊加顯示的 stroke overlay
    格式。"""

    x_position: float
    y_position: float
    width: int
    height: int
    image_base64: str
    inference_ms: float


class ModelStatus(BaseModel):
    loaded: bool
    device: str
    version: str


class HealthResponse(BaseModel):
    status: str
    models: Dict[str, ModelStatus]
