from multiprocessing import Process
from transformers import AutoModel
import os

os.environ["HF_HUB_OFFLINE"] = "0"

def load(model_name):
    print(f"📦 正在预加载: {model_name}")
    AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir="/runpod-volume/hub")
    print(f"✅ 预加载完成: {model_name}")

if __name__ == "__main__":
    models = [
        "BAAI/bge-large-zh-v1.5",
        "Marqo/marqo-fashionCLIP"
    ]
    processes = [Process(target=load, args=(name,)) for name in models]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("🚀 所有模型预加载完成")
