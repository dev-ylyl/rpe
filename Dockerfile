ARG WORKER_CUDA_VERSION=12.6.2
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

# 安装系统依赖和工具
RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV HF_HOME=/runpod-volume \
    U2NET_HOME=/runpod-volume/rembg \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# 安装 Python 依赖
COPY requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# ✅ 下载 rembg ONNX 模型（提前缓存，避免冷启动时联网）
RUN mkdir -p /runpod-volume/rembg && \
    wget -O /runpod-volume/rembg/u2net.onnx https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net.onnx && \
    wget -O /runpod-volume/rembg/isnet-general-use.onnx https://huggingface.co/martintomov/comfy/resolve/8505d94ccac0a7a3dd6e779e0db27ab37ee7004a/rembg/isnet-general-use.onnx

# ✅ 构建时预加载模型（使用多进程）
COPY scripts/preload_models.py /app/scripts/preload_models.py
RUN python3.11 /app/scripts/preload_models.py

# 复制代码并设置工作目录
COPY app /app
WORKDIR /app

# 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]