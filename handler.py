import runpod
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from rembg import remove, new_session
from PIL import Image
import torch, base64, io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# 模型加载
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True, cache_dir="/runpod-volume/hub")
text_model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True, cache_dir="/runpod-volume/hub").cuda().eval()
image_model = AutoModel.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True, cache_dir="/runpod-volume/hub").cuda().eval()
image_processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True, cache_dir="/runpod-volume/hub")
rembg_session = new_session("isnet-general-use")

import traceback

def handler(job):
    logging.info(f"📥 接收到任务: {job}")
    try:
        body = job["input"]
        model_type = body.get("model", "text-embedding")
        inputs = body.get("input")
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