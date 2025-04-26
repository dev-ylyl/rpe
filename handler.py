import runpod
from rp_response import runpod_response
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from rembg import remove, new_session
from PIL import Image
import torch
import base64
import io
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ 加载模型（全部从本地路径，强制不联网）
tokenizer = AutoTokenizer.from_pretrained(
    "/runpod-volume/hub/models--BAAI--bge-large-zh-v1.5",
    trust_remote_code=True,
    local_files_only=True
)

text_model = AutoModel.from_pretrained(
    "/runpod-volume/hub/models--BAAI--bge-large-zh-v1.5",
    trust_remote_code=True,
    local_files_only=True
).cuda().eval()

image_model = AutoModel.from_pretrained(
    "/runpod-volume/hub/models--Marqo--marqo-fashionCLIP",
    trust_remote_code=True,
    local_files_only=True
).cuda().eval()

image_processor = AutoProcessor.from_pretrained(
    "/runpod-volume/hub/models--Marqo--marqo-fashionCLIP",
    trust_remote_code=True,
    local_files_only=True
)

rembg_session = new_session("isnet-general-use")

# ✅ 核心处理函数
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
            return runpod_response(
                status_code=400,
                content_type="application/json",
                body={"error": "Empty input provided.", "model": model_type}
            )

        results = []

        if model_type == "text-embedding":
            logging.info(f"🔠 文本嵌入处理，数量: {len(inputs)}")
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
            logging.info(f"🖼️ 图像嵌入处理，数量: {len(inputs)}")
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

        return runpod_response(
            status_code=200,
            content_type="application/json",
            body={
                "object": "list",
                "data": results,
                "model": model_type,
                "usage": {
                    "prompt_tokens": len(inputs),
                    "total_tokens": len(inputs)
                }
            }
        )

    except Exception as e:
        logging.error(f"❌ 出现异常: {str(e)}")
        traceback.print_exc()
        return runpod_response(
            status_code=500,
            content_type="application/json",
            body={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )

# ✅ 启动 Serverless Worker
logging.info("🟢 Worker 已启动，等待任务中...")
runpod.serverless.start({"handler": handler})