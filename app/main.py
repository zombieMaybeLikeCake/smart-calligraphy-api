import io
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app import inference
from app.config import settings
from app.schemas import HealthResponse, ModelStatus, StrokeResult

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 5 * 1024 * 1024  # 5 MB

# word/stroke 是各自獨立的端點，不是同一個模型的參數切換，服務啟動時
# 兩個都要載入常駐。用 dict 存兩個 ModelHandle，不是模組層級單一變數
# ——這就是階段一把 inference.py 從全域單例改成 ModelHandle 的原因。
_handles: Dict[str, inference.ModelHandle] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時各載入一次，不是每個請求都重新載入
    _handles["word"] = inference.load_model(
        model_dir=settings.word.dir,
        epoch=settings.word.epoch,
        device=settings.device,
        embedding_num=settings.embedding_num,
        embedding_dim=settings.embedding_dim,
        version=settings.word.version,
    )
    _handles["stroke"] = inference.load_model(
        model_dir=settings.stroke.dir,
        epoch=settings.stroke.epoch,
        device=settings.device,
        embedding_num=settings.embedding_num,
        embedding_dim=settings.embedding_dim,
        version=settings.stroke.version,
    )
    yield
    _handles.clear()


app = FastAPI(
    title="Smart Calligraphy Practice API",
    description="書法字跡風格轉換推論服務 —— /predict/word 整字、/predict/stroke 單一筆劃",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models={
            name: ModelStatus(loaded=True, device=handle.device, version=handle.version)
            for name, handle in _handles.items()
        },
    )


async def _decode_upload(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"僅支援 {sorted(ALLOWED_TYPES)}，收到 {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_SIZE:
        raise HTTPException(413, "檔案超過 5 MB 上限")

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "無法解析為圖片")


def _run_predict(
    task: str,
    image: Image.Image,
    label: int,
    red: int,
    green: int,
    blue: int,
    alph: int,
) -> StrokeResult:
    handle = _handles.get(task)
    if handle is None:
        raise HTTPException(503, f"{task} 模型尚未載入")

    if not 0 <= label < settings.embedding_num:
        raise HTTPException(400, f"label 必須介於 0 ~ {settings.embedding_num - 1}")

    start = time.perf_counter()
    try:
        result = inference.predict(
            handle, image, label=label, stroke_color=(red, green, blue, alph)
        )
    except ValueError as exc:
        # fitbest() 在圖片裡偵測不到筆劃時丟這個——是使用者輸入問題，不是伺服器錯誤
        raise HTTPException(400, str(exc))
    elapsed = (time.perf_counter() - start) * 1000

    return StrokeResult(**result, inference_ms=round(elapsed, 2))


@app.post("/predict/word", response_model=StrokeResult)
async def predict_word(
    file: UploadFile = File(...),
    label: int = Query(0, description="風格類別索引，0 ~ embedding_num-1"),
    red: int = Query(0, ge=0, le=255, description="筆劃上色的 R"),
    green: int = Query(0, ge=0, le=255, description="筆劃上色的 G"),
    blue: int = Query(0, ge=0, le=255, description="筆劃上色的 B"),
    alph: int = Query(255, ge=0, le=255, description="筆劃上色的 Alpha"),
):
    """整字風格轉換：上傳一張手寫字圖片，回傳生成結果。"""
    image = await _decode_upload(file)
    return _run_predict("word", image, label, red, green, blue, alph)


@app.post("/predict/stroke", response_model=StrokeResult)
async def predict_stroke(
    file: UploadFile = File(...),
    label: int = Query(0, description="風格類別索引，0 ~ embedding_num-1"),
    red: int = Query(0, ge=0, le=255, description="筆劃上色的 R"),
    green: int = Query(0, ge=0, le=255, description="筆劃上色的 G"),
    blue: int = Query(0, ge=0, le=255, description="筆劃上色的 B"),
    alph: int = Query(255, ge=0, le=255, description="筆劃上色的 Alpha"),
):
    """單一筆劃風格轉換。目前這組 checkpoint（epoch 3000）品質較弱，見
    app/config.py 裡的說明，回應格式跟 /predict/word 完全一樣。"""
    image = await _decode_upload(file)
    return _run_predict("stroke", image, label, red, green, blue, alph)
