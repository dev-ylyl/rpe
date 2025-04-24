import os
from transformers import AutoModel

os.environ["HF_HUB_OFFLINE"] = "0"

for attempt in range(3):
    try:
        AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True)
        AutoModel.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True)
        print("✅ 模型预加载成功")
        break
    except Exception as e:
        print(f"❌ 模型下载失败 attempt {attempt+1}: {e}")
        if attempt == 2:
            raise