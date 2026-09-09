# ---------- builder ----------
# 底圖用純 CUDA runtime（沒有預裝 torch），所有 Python 套件都透過
# requirements.txt 用 pip 裝——這樣才能保證容器裡的 torch 版本，
# 跟本機 conda env 裡實際驗證過 app/main.py 能跑的那組完全一致，
# 不會有「base image 自己內建一份 conda 版 torch、pip 又裝一份」
# 兩邊版本/來源對不上的風險。
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3-pip python3-distutils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime ----------
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# libglib2.0-0：Pillow/常見影像處理常缺的系統相依
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.8 /usr/local/bin/python

COPY --from=builder /install /usr/local

# 非 root 執行：先建帳號，COPY 時直接用 --chown 落地為 appuser 所有，
# 不要「COPY 完再 chown -R」——對已經有 ~500MB checkpoint/套件的目錄做
# chown -R 會把整個目錄的內容重寫進新的一層，實測讓 image 白白多 495MB。
RUN useradd -m appuser

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser model/ ./model/
COPY --chown=appuser:appuser utils/ ./utils/
COPY --chown=appuser:appuser fonts/ ./fonts/
COPY --chown=appuser:appuser models/ ./models/

ENV DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # pip（在 builder 階段用 --prefix=/install 裝）把套件放進
    # .../site-packages，但 Debian/Ubuntu 系統 Python 只認自己 patch 過的
    # .../dist-packages，預設 sys.path 找不到 site-packages，實測會直接
    # ModuleNotFoundError（先前 build 撞過：/usr/local/bin/python: No
    # module named uvicorn）。明講 PYTHONPATH 是最直接、最不會被下一次
    # 改動悄悄弄壞的修法。
    PYTHONPATH=/usr/local/lib/python3.8/site-packages

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
