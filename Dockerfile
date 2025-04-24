ARG WORKER_CUDA_VERSION=11.8.0
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV HF_HOME=/runpod-volume \
    U2NET_HOME=/runpod-volume/rembg \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# 安装Python依赖（带CUDA的torch + transformers等）
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt --no-cache-dir

# 预下载 transformers 模型（容错）
RUN python -c "\
import os; \
os.environ['HF_HUB_OFFLINE'] = '0'; \
from transformers import AutoModel; \
for attempt in range(3): \
    try: \
        AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5', local_files_only=attempt>0); \
        AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', local_files_only=attempt>0); \
        break; \
    except Exception as e: \
        print(f'模型下载尝试 {attempt+1} 失败: {str(e)}'); \
        if attempt == 2: raise"

# 预下载 rembg 模型（isnet-general-use 和 u2net）
RUN mkdir -p /runpod-volume/rembg && \
    wget -O /runpod-volume/rembg/isnet-general-use.onnx https://huggingface.co/ckpt/rembg/resolve/main/isnet-general-use.onnx && \
    wget -O /runpod-volume/rembg/u2net.onnx https://huggingface.co/ckpt/rembg/resolve/main/u2net.onnx

# 复制应用代码
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
