from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from rembg import remove, new_session
from PIL import Image
import torch, base64, io
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

app = FastAPI()

# 初始化
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True)
text_model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True).cuda().eval()
image_model = AutoModel.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True).cuda().eval()
image_processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True)
rembg_session = new_session("isnet-general-use")

@app.post("/v1/embeddings")
async def embedding(request: Request):
    body = await request.json()
    model_type = body.get("model", "text-embedding")
    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]

    if model_type == "text-embedding":
        encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = text_model(**encoded).last_hidden_state.mean(dim=1).cpu().tolist()
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(output)],
            "model": model_type,
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}
        }

    elif model_type == "image-embedding":
        embeddings = []
        for img_str in inputs:
            if img_str.startswith("data:image/"):
                img_str = img_str.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(img_str)))
            image = remove(image, session=rembg_session).convert("RGB")
            processed = image_processor(images=image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                vector = image_model.get_image_features(**processed, normalize=True).squeeze().cpu().tolist()
            embeddings.append(vector)

        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": emb} for i, emb in enumerate(embeddings)],
            "model": model_type,
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}
        }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "memory": f"{torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
    }
