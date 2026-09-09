import os
from dataclasses import dataclass, field

# 目前有兩組 checkpoint，共用同一個模型架構
# （input_nc=1 / embedding_num=40 / embedding_dim=128，已驗證過 state_dict 形狀相容）：
#
#   word   —— strokeStytleChangeServerV2.py 原本在用的整字模型，epoch 6000 品質正常。
#   stroke —— smartCalligraphyPractice/README.md 提到的另一組模型，2026-09-09 從
#             使用者的 Google Drive 取得 epoch 3000。已驗證：checkpoint 能正常載入、
#             pipeline 跑得動，但生成結果是模糊色塊，不是清楚筆劃，研判是訓練步數
#             不夠（word 模型要到 6000 才穩定，這組目前只有 3000）。使用者決定先用
#             這組頂著，之後訓練出更好的版本再換掉——這是已知、暫時接受的狀況，
#             不是要在這裡修的 bug。
_MODEL_PRESETS = {
    "word": {
        "dir": "models/checkpoint_word",
        "epoch": 6000,
        "version": "word-epoch-6000",
    },
    "stroke": {
        "dir": "models/checkpoint_stroke",
        "epoch": 3000,
        "version": "stroke-epoch-3000",
    },
}


@dataclass
class Settings:
    # "word"（預設）或 "stroke"，決定下面 model_dir/model_epoch/model_version
    # 用哪一組 preset。想單獨覆寫其中一項，用 MODEL_DIR / MODEL_EPOCH / MODEL_VERSION。
    model_task: str = os.getenv("MODEL_TASK", "word")

    model_dir: str = field(init=False)
    model_epoch: int = field(init=False)
    model_version: str = field(init=False)

    # "cuda"（預設，正式環境）或 "cpu"（開發機/CI 備援）。
    # GPU 不可用時 inference.load_model() 會自動降級並印出警告，不會直接炸掉。
    device: str = os.getenv("DEVICE", "cuda")

    # 模型訓練時的超參數，跟 checkpoint 綁死，不是隨便可調的執行期設定，
    # 目前先寫死成 strokeStytleChangeServerV2.py 裡的原值，word/stroke 兩組共用。
    embedding_num: int = int(os.getenv("EMBEDDING_NUM", "40"))
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "128"))

    # goalword 生成模式用到的字型（見未來 /synthesize 之類端點，目前 /predict 用不到）
    font_path: str = os.getenv("FONT_PATH", "fonts/BiauKai.ttf")

    def __post_init__(self) -> None:
        preset = _MODEL_PRESETS.get(self.model_task, _MODEL_PRESETS["word"])
        self.model_dir = os.getenv("MODEL_DIR", preset["dir"])
        self.model_epoch = int(os.getenv("MODEL_EPOCH", preset["epoch"]))
        self.model_version = os.getenv("MODEL_VERSION", preset["version"])


settings = Settings()
