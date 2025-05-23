import runpod
from transformers import AutoTokenizer, AutoModel
from rembg import remove, new_session
from PIL import Image
import torch
import base64
import io
import logging
import traceback
import time
import open_clip

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
    local_files_only=True,
    torch_dtype=torch.float16
).cuda().eval()

image_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:Marqo/marqo-fashionCLIP',
    precision="fp16"
)
image_model = image_model.cuda().eval()
image_processor = preprocess_val

print(f"image_processor 类型: {type(image_processor)}")
print(f"image_processor.config: {getattr(image_processor, 'config', None)}")

rembg_session = new_session("u2netp")

# 打印当前GPU信息
logging.info(f"🚀 当前使用GPU: {torch.cuda.get_device_name(0)}")

# CUDA 预热 - text_model
with torch.no_grad(), torch.cuda.amp.autocast():
    dummy_inputs = tokenizer(["warmup"], padding=True, return_tensors="pt", truncation=True)
    dummy_inputs = {k: v.cuda() for k, v in dummy_inputs.items()}
    _ = text_model(**dummy_inputs).last_hidden_state.mean(dim=1)
logging.info("✅ 文本模型 warmup 完成")

# CUDA 预热 - image_model
with torch.no_grad(), torch.cuda.amp.autocast():
    dummy_image = Image.new('RGB', (224, 224), color=(255, 255, 255))  # 创建一张白图
    tensor_image = image_processor(dummy_image).unsqueeze(0).cuda()
    _ = image_model.encode_image(tensor_image, normalize=True)
logging.info("✅ 图片模型 warmup 完成")

# ✅ 核心处理函数
def handler(job):
    logging.info(f"📥 任务输入内容:\n{job}\n📄 类型: {type(job)}")
    try:
        model_type = job["input"].get("model", "text")
        inputs = job["input"].get("data")
        logging.info(f"📋 inputs内容是: {inputs} (类型: {type(inputs)}, 长度: {len(inputs) if inputs else 0})")
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            logging.warning("⚠️ 数据为空")
            return {
                "output": {
                    "error": "Empty input provided."
                }
            }

        results = []

        if model_type == "text":
            logging.info(f"🔠 文本嵌入处理，数量: {len(inputs)}")
            start_time = time.time()

            # Tokenizer阶段
            encoded = tokenizer(inputs, padding=True, return_tensors="pt", truncation=True)
            logging.info(f"🧩 Tokenizer输出 keys: {list(encoded.keys())}")

            tokenizer_time = time.time()
            logging.info(f"⏱️ Tokenizer耗时: {tokenizer_time - start_time:.3f}s")

            # 传送到cuda，仅移动，不转半精度
            encoded = {k: v.cuda() for k, v in encoded.items()}
            to_cuda_time = time.time()
            logging.info(f"⏱️ To CUDA耗时: {to_cuda_time - tokenizer_time:.3f}s")

            # 推理阶段
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = text_model(**encoded).last_hidden_state.mean(dim=1).cpu().tolist()

            inference_time = time.time()
            logging.info(f"⏱️ 推理耗时: {inference_time - to_cuda_time:.3f}s")

            for emb in output:
                results.append(emb)

            total_time = time.time()
            logging.info(f"✅ 总处理时间: {total_time - start_time:.3f}s")

        elif model_type == "image":
            logging.info(f"🖼️ 图像嵌入处理，数量: {len(inputs)}")
            start_time = time.time()

            images = []
            for i, img_str in enumerate(inputs):
                img_start_time = time.time()
                if img_str.startswith("data:image/"):
                    img_str = img_str.split(",")[1]
                image = Image.open(io.BytesIO(base64.b64decode(img_str)))
                decode_time = time.time()
                logging.info(f"🖼️ 解码第{i}张图片耗时: {decode_time - img_start_time:.3f}s")

                image = remove(image, session=rembg_session).convert("RGB")
                rembg_time = time.time()
                logging.info(f"🧹 去背景第{i}张图片耗时: {rembg_time - decode_time:.3f}s")

                images.append(image)

            # 批量处理
            try:
                processed_images = torch.stack([image_processor(img) for img in images]).cuda()
            except Exception as e:
                logging.error(f"❌ 图片处理出错: {str(e)}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                return {
                    "output": {
                        "error": f"Image processing error: {str(e)}",
                        "trace": traceback.format_exc()
                    }
                }

            processor_time = time.time()
            logging.info(f"🎛️ 图片批处理耗时: {processor_time - rembg_time:.3f}s")

            with torch.no_grad(), torch.cuda.amp.autocast():
                vectors = image_model.encode_image(processed_images, normalize=True).cpu().tolist()

            inference_time = time.time()
            logging.info(f"⏱️ 图片推理耗时: {inference_time - processor_time:.3f}s")

            for vector in vectors:
                results.append(vector)

            total_time = time.time()
            logging.info(f"✅ 总图片处理时间: {total_time - start_time:.3f}s")

        logging.info(f"✅ 推理完成，共生成 {len(results)} 个embedding，每个embedding维度: {len(results[0])}")
        return {
            "output": {
                "embeddings": results
            }
        }

    except Exception as e:
        logging.error(f"❌ 出现异常: {str(e)}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {
            "output": {
                "error": str(e),
                "trace": traceback.format_exc()
            }
        }

# ✅ 启动 Serverless Worker
logging.info("🟢 Worker 已启动，等待任务中...")
runpod.serverless.start({"handler": handler})