import io
import time
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app import inference
from app.config import settings
from app.schemas import HealthResponse, ModelStatus, StrokeResult, SynthesizeCharResult

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 5 * 1024 * 1024  # 5 MB
# /synthesize/* 每個字都各跑一次推論，不能讓一次請求塞一長串文字就把
# GPU 佔滿、變相繞過速率限制。20 字對展示用途綽綽有餘。
MAX_SYNTHESIZE_CHARS = 20

limiter = Limiter(key_func=get_remote_address)

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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# /demo（下面掛的 StaticFiles）現在跟 API 同源，這個 middleware 本身對它
# 已經不是必要的了，但留著是因為 API 本來就設計成可以被別的網頁/App 打
# （這正是 /predict/* 需要 API Key 而不是靠同源限制擋人的原因），拿掉反而
# 是縮小了服務的用途。allow_origins 開 "*" 是刻意的：CORS 只決定「瀏覽器
# 要不要讓網頁看到回應」，不是存取控制層，收緊它不會讓 API 更安全。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 網頁版 demo：web/index.html 掛在 /demo/ 底下，跟 /predict/* 同一個 process、
# 同一個 origin，不用另外開靜態伺服器。html=True 讓 /demo/ 直接回 index.html。
app.mount("/demo", StaticFiles(directory="web", html=True), name="demo")


def verify_api_key(x_api_key: str = Header(None)):
    """套用在 /predict/* 上，不套在 /health、/docs——服務公開後這兩個
    還是要讓面試官不用金鑰就看得到。settings.api_key 空字串＝不驗證，
    這是本機開發預設；一旦要掛公網，環境變數沒設 API_KEY 就等於忘記關門，
    這點只能自己記得設，程式碼這邊沒辦法幫忙擋。"""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(401, "無效或缺少的 API Key（X-API-Key 標頭）")


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


@app.post(
    "/predict/word", response_model=StrokeResult, dependencies=[Depends(verify_api_key)]
)
@limiter.limit(settings.rate_limit_predict)
async def predict_word(
    request: Request,
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


@app.post(
    "/predict/stroke", response_model=StrokeResult, dependencies=[Depends(verify_api_key)]
)
@limiter.limit(settings.rate_limit_predict)
async def predict_stroke(
    request: Request,
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


def _run_synthesize(
    task: str,
    text: str,
    label: int,
    red: int,
    green: int,
    blue: int,
    alph: int,
) -> List[SynthesizeCharResult]:
    if len(text) > MAX_SYNTHESIZE_CHARS:
        raise HTTPException(400, f"最多一次 {MAX_SYNTHESIZE_CHARS} 個字，收到 {len(text)} 個")

    handle = _handles.get(task)
    if handle is None:
        raise HTTPException(503, f"{task} 模型尚未載入")

    if not 0 <= label < settings.embedding_num:
        raise HTTPException(400, f"label 必須介於 0 ~ {settings.embedding_num - 1}")

    results: List[SynthesizeCharResult] = []
    for ch in text:
        start = time.perf_counter()
        try:
            image = inference.render_char(ch, settings.font_path)
            result = inference.predict(
                handle, image, label=label, stroke_color=(red, green, blue, alph)
            )
        except ValueError as exc:
            # fitbest() 在字型畫出來的圖裡偵測不到筆劃時丟這個
            # （理論上不會發生——字型渲染一定有墨色——但空白字元之類的
            # 邊界情況還是可能觸發，當成使用者輸入問題回 400）
            raise HTTPException(400, f"「{ch}」：{exc}")
        elapsed = (time.perf_counter() - start) * 1000
        results.append(SynthesizeCharResult(char=ch, **result, inference_ms=round(elapsed, 2)))

    return results


@app.post(
    "/synthesize/word",
    response_model=List[SynthesizeCharResult],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_synthesize)
async def synthesize_word(
    request: Request,
    text: str = Query(..., min_length=1, max_length=MAX_SYNTHESIZE_CHARS, description="要生成的文字"),
    label: int = Query(0, description="風格類別索引，0 ~ embedding_num-1"),
    red: int = Query(0, ge=0, le=255, description="筆劃上色的 R"),
    green: int = Query(0, ge=0, le=255, description="筆劃上色的 G"),
    blue: int = Query(0, ge=0, le=255, description="筆劃上色的 B"),
    alph: int = Query(255, ge=0, le=255, description="筆劃上色的 Alpha"),
):
    """打字生成整字風格轉換：不用上傳圖片，文字先用字型（見
    app/config.py 的 font_path）畫成圖，再送進模型。逐字推論，回傳陣列，
    順序對應輸入文字。"""
    return _run_synthesize("word", text, label, red, green, blue, alph)


@app.post(
    "/synthesize/stroke",
    response_model=List[SynthesizeCharResult],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_synthesize)
async def synthesize_stroke(
    request: Request,
    text: str = Query(..., min_length=1, max_length=MAX_SYNTHESIZE_CHARS, description="要生成的文字"),
    label: int = Query(0, description="風格類別索引，0 ~ embedding_num-1"),
    red: int = Query(0, ge=0, le=255, description="筆劃上色的 R"),
    green: int = Query(0, ge=0, le=255, description="筆劃上色的 G"),
    blue: int = Query(0, ge=0, le=255, description="筆劃上色的 B"),
    alph: int = Query(255, ge=0, le=255, description="筆劃上色的 Alpha"),
):
    """打字生成單一筆劃風格轉換，其餘同 /synthesize/word。"""
    return _run_synthesize("stroke", text, label, red, green, blue, alph)
