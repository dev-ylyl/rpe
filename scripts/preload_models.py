import os
import sys
from transformers import AutoModel

os.environ["HF_HUB_OFFLINE"] = "0"

for attempt in range(3):
    try:
        print(f"🚀 第 {attempt + 1} 次尝试加载所有模型...")

        # 先加载第一个
        AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5", trust_remote_code=True)

        # 清理 transformers_modules 缓存（防止路径 hash 冲突）
        sys.modules.pop("transformers_modules", None)

        # 加载第二个
        AutoModel.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True)

        print("✅ 模型预加载成功")
        break
    except Exception as e:
        print(f"❌ 模型下载失败 attempt {attempt+1}: {e}")
        if attempt == 2:
            raise