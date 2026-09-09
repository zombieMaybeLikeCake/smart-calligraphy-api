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

COPY app/ ./app/
COPY model/ ./model/
COPY utils/ ./utils/
COPY fonts/ ./fonts/
COPY models/ ./models/

ENV DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 非 root 執行
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
