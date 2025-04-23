from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import torch
from transformers import TextEmbeddingInferenceModel, AutoProcessor, AutoModel
from rembg import remove
from PIL import Image
import base64
import io
import requests
import numpy as np

app = FastAPI()

# --- TEXT EMBEDDING SETUP ---
text_model = TextEmbeddingInferenceModel.from_pretrained(
    "BAAI/bge-large-zh-v1.5", trust_remote_code=True
)

def get_text_embedding(texts: List[str]) -> List[List[float]]:
    return text_model(texts)

# --- IMAGE EMBEDDING SETUP ---
image_model_name = "Marqo/marqo-fashionCLIP"
image_model = AutoModel.from_pretrained(image_model_name, trust_remote_code=True).cuda()
image_processor = AutoProcessor.from_pretrained(image_model_name, trust_remote_code=True)
image_model.eval()

def remove_background(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    output = remove(img)
    return output.convert("RGB")

def get_image_embedding(image_b64: str) -> List[float]:
    header, encoded = image_b64.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))
    image = remove_background(image)
    processed = image_processor(images=image, return_tensors="pt")
    pixel_values = processed["pixel_values"].cuda()
    with torch.no_grad():
        embedding = image_model.get_image_features(pixel_values=pixel_values, normalize=True)
    return embedding.squeeze().cpu().tolist()

# --- OpenAI API-compatible interface ---
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

@app.post("/v1/embeddings")
def create_embedding(req: EmbeddingRequest):
    if isinstance(req.input, str):
        inputs = [req.input]
    else:
        inputs = req.input

    if req.model == "text":
        embeddings = get_text_embedding(inputs)
    elif req.model == "image":
        embeddings = [get_image_embedding(img) for img in inputs]
    else:
        raise HTTPException(status_code=400, detail="Model must be 'text' or 'image'")

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": emb
            } for i, emb in enumerate(embeddings)
        ],
        "model": req.model,
        "usage": {
            "prompt_tokens": len(inputs),
            "total_tokens": len(inputs)
        }
    }
