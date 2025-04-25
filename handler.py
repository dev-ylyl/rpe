import runpod
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from rembg import remove, new_session
from PIL import Image
import torch, base64, io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

from huggingface_hub import snapshot_download
import os

try:
    tokenizer_snapshot = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir="/runpod-volume/hub", local_files_only=True)
    logging.info(f"✅ tokenizer 缓存命中路径: {tokenizer_snapshot}")
except Exception:
    logging.warning("⚠️ tokenizer 未命中缓存，将使用在线加载（可能触发网络请求）")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True, cache_dir="/runpod-volume/hub")

try:
    text_model_snapshot = snapshot_download("BAAI/bge-large-zh-v1.5", cache_dir="/runpod-volume/hub", local_files_only=True)
    logging.info(f"✅ text_model 缓存命中路径: {text_model_snapshot}")
except Exception:
    logging.warning("⚠️ text_model 未命中缓存，将使用在线加载（可能触发网络请求）")
text_model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True, cache_dir="/runpod-volume/hub").cuda().eval()

try:
    image_model_snapshot = snapshot_download("Marqo/marqo-fashionCLIP", cache_dir="/runpod-volume/hub", local_files_only=True)
    logging.info(f"✅ image_model 缓存命中路径: {image_model_snapshot}")
except Exception:
    logging.warning("⚠️ image_model 未命中缓存，将使用在线加载（可能触发网络请求）")
image_model = AutoModel.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True, cache_dir="/runpod-volume/hub").cuda().eval()

try:
    image_processor_snapshot = snapshot_download("Marqo/marqo-fashionCLIP", cache_dir="/runpod-volume/hub", local_files_only=True)
    logging.info(f"✅ image_processor 缓存命中路径: {image_processor_snapshot}")
except Exception:
    logging.warning("⚠️ image_processor 未命中缓存，将使用在线加载（可能触发网络请求）")
image_processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True, cache_dir="/runpod-volume/hub")
rembg_session = new_session("isnet-general-use")

import traceback

def handler(job):
    logging.info(f"📥 接收到任务: {job}")
    try:
        openai_input = job["input"].get("openai_input", {})
        model_type = openai_input.get("model", "text-embedding")
        inputs = openai_input.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            logging.warning("⚠️ 输入为空")
            result = {
                "error": "Empty input provided.",
                "model": model_type
            }
            logging.info(f"✅ 返回结果: {result}")
            return result

        results = []

        if model_type == "text-embedding":
            logging.info(f"🔠 处理文本嵌入，输入数量: {len(inputs)}")
            encoded = tokenizer(text=inputs, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                output = text_model(**encoded).last_hidden_state.mean(dim=1).cpu().tolist()
            for i, emb in enumerate(output):
                results.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": emb
                })

        elif model_type == "image-embedding":
            logging.info(f"🖼️ 处理图像嵌入，图片数量: {len(inputs)}")
            for i, img_str in enumerate(inputs):
                if img_str.startswith("data:image/"):
                    img_str = img_str.split(",")[1]
                image = Image.open(io.BytesIO(base64.b64decode(img_str)))
                image = remove(image, session=rembg_session).convert("RGB")
                processed = image_processor(images=image, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    vector = image_model.get_image_features(**processed, normalize=True).squeeze().cpu().tolist()
                results.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": vector
                })

        result = {
            "object": "list",
            "data": results,
            "model": model_type,
            "usage": {
                "prompt_tokens": len(inputs),
                "total_tokens": len(inputs)
            }
        }
        logging.info(f"✅ 返回结果: {result}")
        logging.info("🚀 任务处理完成，无异常抛出，正常返回结果。")
        return result

    except Exception as e:
        logging.error(f"❌ 出现异常: {str(e)}")
        traceback.print_exc()
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

logging.info("🟢 Worker 已启动，等待任务中...")
runpod.serverless.start({"handler": handler})