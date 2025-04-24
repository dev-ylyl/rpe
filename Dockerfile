# 继承官方参数化设计，但默认使用更稳定的CUDA 11.8
ARG WORKER_CUDA_VERSION=11.8.0
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

# 安装依赖（兼容Python 3.9-3.11）
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall torch -y && \
    CUDA_VERSION_SHORT=$(echo ${WORKER_CUDA_VERSION} | cut -d. -f1,2 | tr -d .) && \
    pip install torch==2.1.0+cu${CUDA_VERSION_SHORT} \
    --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}

# 预下载模型到持久化目录（BAAI + Marqo + Rembg自动下载）
ENV HF_HOME=/runpod-volume \
    U2NET_HOME=/runpod-volume/rembg
RUN python -c "
from transformers import AutoModel; \
AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5'); \
AutoModel.from_pretrained('Marqo/marqo-fashionCLIP')
"

# 复制代码（包含优化后的main.py）
COPY . .

# 启动命令（FastAPI替代原handler）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]