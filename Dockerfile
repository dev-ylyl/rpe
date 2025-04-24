ARG WORKER_CUDA_VERSION=12.6.2
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

ARG WORKER_CUDA_VERSION=12.6.2

# 安装基础依赖
RUN apt-get update -o Acquire::Retries=5 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 wget && \
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

# 安装 PyTorch 2.7.0 稳定版（支持 CUDA 12.6）
RUN pip uninstall torch -y && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 \
        --no-cache-dir && \
    pip install hf_transfer open_clip_torch

# 预下载 transformers 模型（容错3次）
COPY scripts/preload_models.py ./preload_models.py
RUN python3.11 preload_models.py

# 下载 rembg 模型（双版本）
RUN mkdir -p /runpod-volume/rembg && \
    wget -O /runpod-volume/rembg/u2net.onnx https://huggingface.co/ckpt/rembg/resolve/main/u2net.onnx && \
    wget -O /runpod-volume/rembg/isnet-general-use.onnx https://huggingface.co/ckpt/rembg/resolve/main/isnet-general-use.onnx

# 拷贝应用代码
COPY . .

# 启动服务
CMD ["python3.11", "-u", "main.py"]