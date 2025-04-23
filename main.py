from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from rembg import remove
from PIL import Image
import base64
import io
import requests
import numpy as np

app = FastAPI()

# --- TEXT EMBEDDING SETUP ---
text_model_name = "BAAI/bge-large-zh-v1.5"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name).cuda()
text_model.eval()

def get_text_embedding(texts: List[str]) -> List[List[float]]:
    inputs = text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.cpu().tolist()

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
