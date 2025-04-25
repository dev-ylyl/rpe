import runpod
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from rembg import remove, new_session
from PIL import Image
import torch, base64, io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# 模型加载
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True)
text_model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True).cuda().eval()
image_model = AutoModel.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True).cuda().eval()
image_processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True)
rembg_session = new_session("isnet-general-use")

def handler(job):
    body = job["input"]
    model_type = body.get("model", "text-embedding")
    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]

    if model_type == "text-embedding":
        encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = text_model(**encoded).last_hidden_state.mean(dim=1).cpu().tolist()
        return [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(output)]

    elif model_type == "image-embedding":
        results = []
        for i, img_str in enumerate(inputs):
            if img_str.startswith("data:image/"):
                img_str = img_str.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(img_str)))
            image = remove(image, session=rembg_session).convert("RGB")
            processed = image_processor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                vector = image_model.get_image_features(**processed, normalize=True).squeeze().cpu().tolist()
            results.append({"object": "embedding", "index": i, "embedding": vector})
        return results

runpod.serverless.start({"handler": handler})