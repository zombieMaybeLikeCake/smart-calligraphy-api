import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Zi2ziModel.load_networks(epoch) 找的是 f"{epoch}_net_G.pth" / f"{epoch}_net_D.pth"，
    # 所以模型設定是「目錄 + epoch」，不是單一檔案路徑。
    model_dir: str = os.getenv("MODEL_DIR", "models/checkpoint")
    model_epoch: int = int(os.getenv("MODEL_EPOCH", "6000"))
    model_version: str = os.getenv("MODEL_VERSION", "epoch-6000")

    # "cuda"（預設，正式環境）或 "cpu"（開發機/CI 備援）。
    # GPU 不可用時 inference.load_model() 會自動降級並印出警告，不會直接炸掉。
    device: str = os.getenv("DEVICE", "cuda")

    # 模型訓練時的超參數，跟 checkpoint 綁死，不是隨便可調的執行期設定，
    # 目前先寫死成 strokeStytleChangeServerV2.py 裡的原值。
    embedding_num: int = int(os.getenv("EMBEDDING_NUM", "40"))
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "128"))

    # goalword 生成模式用到的字型（見未來 /synthesize 之類端點，目前 /predict 用不到）
    font_path: str = os.getenv("FONT_PATH", "fonts/BiauKai.ttf")


settings = Settings()
