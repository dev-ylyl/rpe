from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModel
from rembg import remove, new_session
from PIL import Image
import base64
import io
import numpy as np
from contextlib import asynccontextmanager
import logging

# 配置日志直接输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """冷启动优化：预加载模型并显示显存占用"""
    logging.info("🚀 初始化模型中...")
    
    # 文本模型（BAAI）
    app.state.text_model = AutoModel.from_pretrained(
        "BAAI/bge-large-zh-v1.5",
        trust_remote_code=True
    ).cuda().eval()
    
    # 图像模型（Marqo）
    app.state.image_model = AutoModel.from_pretrained(
        "Marqo/marqo-fashionCLIP",
        trust_remote_code=True
    ).cuda().eval()
    
    # Rembg会话
    app.state.rembg_session = new_session("isnet-general-use")
    
    logging.info(f"✅ 初始化完成 | 显存占用: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    yield
    
    # 清理GPU内存
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    input: str | List[str]  # 支持OpenAI格式的字符串或数组
    model: str = "text-embedding"  # 可选 "text-embedding" 或 "image-embedding"
    user: str = None  # 兼容OpenAI字段

def process_image(image_b64: str, session) -> Image.Image:
    """Base64图像处理流水线"""
    try:
        # 兼容纯Base64和DataURL格式
        if image_b64.startswith('data:image/'):
            image_b64 = image_b64.split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        return remove(img, session=session).convert("RGB")
    except Exception as e:
        logging.error(f"图像处理失败: {str(e)}")
        raise

@app.post("/v1/embeddings")
async def create_embedding(request: Request):
    """OpenAI兼容的嵌入端点"""
    try:
        inputs = [request.input] if isinstance(request.input, str) else request.input
        
        # 文本处理
        if request.model == "text-embedding":
            logging.info(f"📝 处理文本输入（长度: {len(inputs[0])}）")
            with torch.no_grad():
                tokenized = app.state.text_model.tokenize(inputs)
                embeddings = app.state.text_model(**tokenized).last_hidden_state.mean(dim=1).tolist()
        
        # 图像处理
        elif request.model == "image-embedding":
            logging.info(f"🖼️ 处理图像输入（数量: {len(inputs)}）")
            embeddings = []
            for img_str in inputs:
                img = process_image(img_str, app.state.rembg_session)
                with torch.no_grad():
                    # 极简归一化 (替代processor)
                    img_tensor = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float().cuda()
                    img_tensor = (img_tensor - 127.5) / 127.5
                    embedding = app.state.image_model(img_tensor)[0].tolist()
                    embeddings.append(embedding)
        
        # 构建OpenAI格式响应
        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": i,
                "embedding": emb
            } for i, emb in enumerate(embeddings)],
            "model": request.model,
            "usage": {
                "prompt_tokens": len(inputs),
                "total_tokens": len(inputs)
            }
        }
    except Exception as e:
        logging.error(f"❌ 请求处理失败: {str(e)}")
        return {
            "error": {
                "message": str(e),
                "type": "invalid_request_error"
            }
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "memory_used": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB"
    }