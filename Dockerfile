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

# 安装Python依赖
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip uninstall torch -y && \
    CUDA_VERSION_SHORT=$(echo ${WORKER_CUDA_VERSION} | cut -d. -f1,2 | tr -d .) && \
    pip install torch==2.1.0+cu${CUDA_VERSION_SHORT} \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}

# 预下载模型（修正后的多行命令写法）
RUN python -c "\
from transformers import AutoModel; \
AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5'); \
AutoModel.from_pretrained('Marqo/marqo-fashionCLIP')"

# 复制应用代码
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]