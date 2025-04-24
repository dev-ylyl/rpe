# Dockerfile (加入 USER 和权限优化)

ARG WORKER_CUDA_VERSION=12.6.2
FROM runpod/base:0.6.2-cuda${WORKER_CUDA_VERSION}

ARG WORKER_CUDA_VERSION=12.6.2 # 再次声明 ARG

# 设置 Python 版本
ENV PYTHON_VERSION=3.11

# 1. 安装基础依赖 (以 root 身份)
RUN apt-get update -o Acquire::Retries=5 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 wget \
    # 确保安装了 python 和 pip
    python${PYTHON_VERSION} python${PYTHON_VERSION}-pip python${PYTHON_VERSION}-venv && \
    # 清理 apt 缓存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. 设置环境变量 (以 root 身份)
ENV HF_HOME=/runpod-volume \
    U2NET_HOME=/runpod-volume/rembg \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # 明确设置缓存路径 (可选，但清晰)
    TRANSFORMERS_CACHE=${HF_HOME}/hub \
    HF_HUB_CACHE=${HF_HOME}/hub \
    HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub \
    # 更新 PATH，包含用户安装的包路径 (假定用户为 runpod 或标准 1000 用户)
    # /home/runpod/.local/bin 是 pip install --user 的常见路径
    PATH="/usr/bin/python${PYTHON_VERSION}:/home/runpod/.local/bin:/home/user/.local/bin:~/.local/bin:${PATH}"

# 3. 创建缓存目录并设置权限 (以 root 身份, 在切换用户前)
# RunPod Serverless 通常以 UID 1000 运行，请根据实际情况调整
# 创建 HF 和 Rembg 的缓存目录
RUN mkdir -p ${HF_HOME}/hub ${HF_HOME}/modules ${HF_HOME}/assets ${U2NET_HOME} && \
    # 将这些目录的所有权交给目标用户 (UID 1000, GID 1000)
    chown -R 1000:1000 ${HF_HOME} ${U2NET_HOME}

# 4. 切换到非 root 用户 (关键步骤!)
# 假设 RunPod Serverless 运行时用户是 UID 1000, GID 1000
USER 1000:1000

# 5. 设置工作目录 (作为非 root 用户)
WORKDIR /app

# 6. 复制并安装 Python 依赖 (作为非 root 用户)
# 复制 requirements.txt，确保所有权正确
COPY --chown=1000:1000 requirements.txt /app/requirements.txt
# 在安装 requirements 时指定 index url for torch (cu126)
RUN python${PYTHON_VERSION} -m pip install --upgrade pip && \
    python${PYTHON_VERSION} -m pip install --no-cache-dir \
        # 使用此 index URL 来查找 requirements.txt 中的 torch==2.7.0
        --index-url https://download.pytorch.org/whl/cu126 \
        -r /app/requirements.txt && \
    rm /app/requirements.txt

# 7. 安装特定版本 PyTorch (作为非 root 用户)
# 注意: 如果 torch, torchvision, torchaudio 已在 requirements.txt 中固定版本，则此步骤多余，可以注释掉。
# 警告: 请务必确认 PyTorch 版本与 CUDA 版本的组合是否有 'cu126' 的 wheel 文件。
# 如果 'cu126' 不可用或不稳定，请根据 PyTorch 官网查找适合 CUDA ${WORKER_CUDA_VERSION} 的正确后缀 (如 'cu121')。
# RUN python${PYTHON_VERSION} -m pip uninstall torch torchvision torchaudio -y || true # 允许失败
# RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
#     torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cu126 # <--- 仔细验证此 URL！如果构建失败或运行时出错，尝试 cu121

# 安装 hf_transfer (作为非 root 用户，如果 requirements.txt 中没有)
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir hf_transfer

# 8. 复制并运行预下载脚本 (作为非 root 用户)
# 复制 scripts 目录，确保所有权
COPY --chown=1000:1000 scripts/ /app/scripts/
# 以非 root 用户身份运行脚本，写入到已授权的 HF_HOME 目录
RUN python${PYTHON_VERSION} /app/scripts/preload_models.py

# 9. 下载 rembg 模型 (作为非 root 用户)
# mkdir -p 已在步骤 3 完成，这里只需下载
# wget 会写入到 U2NET_HOME，该目录已授权给用户 1000
RUN wget -O ${U2NET_HOME}/u2net.onnx https://huggingface.co/ckpt/rembg/resolve/main/u2net.onnx && \
    wget -O ${U2NET_HOME}/isnet-general-use.onnx https://huggingface.co/ckpt/rembg/resolve/main/isnet-general-use.onnx

# 10. 复制应用代码 (作为非 root 用户)
# 将当前目录所有内容复制到 /app，确保所有权
COPY --chown=1000:1000 . /app

# 11. 暴露端口 (FastAPI 默认 8000)
EXPOSE 8000

# 12. 启动服务 (作为非 root 用户)
# 使用 uvicorn 启动 FastAPI 应用
# 确保 uvicorn 已通过 requirements.txt 安装
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]