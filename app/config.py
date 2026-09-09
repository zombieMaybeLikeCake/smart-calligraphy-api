import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    dir: str
    epoch: int
    version: str


@dataclass
class Settings:
    # "cuda"（預設，正式環境）或 "cpu"（開發機/CI 備援）。
    # GPU 不可用時 inference.load_model() 會自動降級並印出警告，不會直接炸掉。
    device: str = os.getenv("DEVICE", "cuda")

    # 模型訓練時的超參數，跟 checkpoint 綁死，不是隨便可調的執行期設定，
    # 目前先寫死成 strokeStytleChangeServerV2.py 裡的原值，word/stroke 兩組共用同一種架構。
    embedding_num: int = int(os.getenv("EMBEDDING_NUM", "40"))
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "128"))

    # /synthesize/* 用來把打的字轉成圖片的字型，/predict/* 用不到。
    font_path: str = os.getenv("FONT_PATH", "fonts/BiauKai.ttf")

    # 空字串＝不驗證（本機開發預設）。服務一旦掛上公網（見 README 的 frp 那節）
    # 一定要設這個，不然任何人都能打 /predict/*。真正的 key 只從環境變數讀，
    # 不進 git、不寫死在程式碼裡。
    api_key: str = os.getenv("API_KEY", "")

    # 每個來源 IP 對 /predict/* 的速率限制，slowapi 的字串格式（"10/minute" 等）。
    # 注意：透過 frp tcp 模式轉發時，FastAPI 看到的來源 IP 是 frps 的位址，
    # 不是真正的訪客，這會讓「每 IP」限流退化成「所有人共用同一個額度」。
    # 要修正這個要嘛在 frpc 那條 proxy 開 transport.proxyProtocolVersion = "v2"
    # 並在 uvicorn 加 --proxy-headers，要嘛接受這個限制、當作全域限流來看。
    # 目前選後者（doesn't block，先求有再求精確），frp 設定之後想改再改。
    rate_limit_predict: str = os.getenv("RATE_LIMIT_PREDICT", "10/minute")

    # /synthesize/* 一次請求可能對每個字各跑一次推論（見 main.py 的
    # MAX_SYNTHESIZE_CHARS），比 /predict/* 單張圖片重得多，所以限制設得更緊，
    # 不然一次打長字串就能繞過 /predict/* 那組限流的用意。
    rate_limit_synthesize: str = os.getenv("RATE_LIMIT_SYNTHESIZE", "5/minute")

    # /predict/word、/predict/stroke 是各自獨立的端點，服務啟動時兩個模型都要載入，
    # 不是二選一，所以這裡兩組設定同時存在，不是像階段一那樣用 MODEL_TASK 切換。
    #
    #   word   —— strokeStytleChangeServerV2.py 原本在用的整字模型，epoch 6000 品質正常。
    #   stroke —— smartCalligraphyPractice/README.md 提到的另一組模型，2026-09-09 從
    #             使用者的 Google Drive 取得 epoch 3000。已驗證：checkpoint 能正常載入、
    #             pipeline 跑得動，但生成結果是模糊色塊，研判是訓練步數不夠（word 模型
    #             要到 6000 才穩定，這組目前只有 3000）。使用者決定先用這組頂著，之後
    #             訓練出更好的版本再換掉——這是已知、暫時接受的狀況，不是要在這裡修的 bug。
    word: ModelConfig = ModelConfig(
        dir=os.getenv("WORD_MODEL_DIR", "models/checkpoint_word"),
        epoch=int(os.getenv("WORD_MODEL_EPOCH", "6000")),
        version=os.getenv("WORD_MODEL_VERSION", "word-epoch-6000"),
    )
    stroke: ModelConfig = ModelConfig(
        dir=os.getenv("STROKE_MODEL_DIR", "models/checkpoint_stroke"),
        epoch=int(os.getenv("STROKE_MODEL_EPOCH", "3000")),
        version=os.getenv("STROKE_MODEL_VERSION", "stroke-epoch-3000"),
    )


settings = Settings()
