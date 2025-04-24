ARG WORKER_CUDA_VERSION=11.8.0
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV HF_HOME=/runpod-volume \
    U2NET_HOME=/runpod-volume/rembg \
    PIP_NO_CACHE_DIR=1

# 安装Python依赖（单步完成）
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir && \
    pip uninstall torch -y && \
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 预下载模型
RUN python -c "\
from transformers import AutoModel; \
AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5'); \
AutoModel.from_pretrained('Marqo/marqo-fashionCLIP')"

# 复制应用代码
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]