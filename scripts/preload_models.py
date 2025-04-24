# scripts/preload_models.py
import os
import sys
import shutil
import torch  # 用于清理 GPU 缓存
from transformers import AutoModel, AutoConfig, AutoProcessor, AutoTokenizer
import logging
import time

# --- 配置 ---
# 配置日志记录器，输出到标准输出，方便 Docker build 查看
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PreloadModels")

# 要预加载的模型列表:
#   name: Hugging Face Hub 上的模型名称
#   has_tokenizer: 是否需要预加载 Tokenizer
#   has_processor: 是否需要预加载 Processor
MODELS_TO_PRELOAD = [
    {"name": "BAAI/bge-large-zh-v1.5", "has_tokenizer": True, "has_processor": False},
    {"name": "Marqo/marqo-fashionCLIP", "has_tokenizer": False, "has_processor": True}
]
MAX_ATTEMPTS = 3  # 整个预加载过程的最大重试次数

# 从环境变量获取 HF_HOME，这必须与 Dockerfile 中的 ENV HF_HOME 一致！
HF_HOME = os.environ.get("HF_HOME")
if not HF_HOME:
    logger.error("关键错误: HF_HOME 环境变量未设置! 无法确定缓存目录。")
    logger.error("请确保 Dockerfile 中设置了 'ENV HF_HOME=/runpod-volume' (或类似的持久路径)")
    sys.exit(1) # 必须失败退出

# 根据 HF_HOME 定义缓存路径
MODULES_CACHE = os.path.join(HF_HOME, "modules") # 存放 trust_remote_code=True 下载的代码
HUB_CACHE = os.path.join(HF_HOME, "hub")       # 存放模型权重、配置等主要文件
# --- 配置结束 ---

def clear_dynamic_modules_cache():
    """
    **核心步骤**: 清理为 `trust_remote_code=True` 动态加载的代码缓存。
    """
    logger.info("--- 正在清理动态代码缓存 (`modules` 目录和 `sys.modules`) ---")

    # 1. 清理文件系统缓存 (`modules` 目录)
    if os.path.exists(MODULES_CACHE):
        logger.info(f"尝试删除 Transformers 模块缓存目录: {MODULES_CACHE}")
        try:
            shutil.rmtree(MODULES_CACHE, ignore_errors=True)
            time.sleep(0.5) # 短暂等待
            if not os.path.exists(MODULES_CACHE):
                logger.info("模块缓存目录已成功删除。")
            else:
                logger.warning("删除模块缓存目录后其仍然存在 (权限问题或文件被锁定?)")
        except Exception as e:
            logger.warning(f"删除模块缓存目录时发生错误: {e}")
    else:
        logger.info("模块缓存目录不存在，跳过删除。")

    # 2. 清理内存中已加载的 Python 模块 (`sys.modules`)
    keys_to_delete = [k for k in sys.modules if k.startswith('transformers_modules')]
    if keys_to_delete:
        logger.info(f"正在从 sys.modules 中移除 {len(keys_to_delete)} 个动态加载的 'transformers_modules.*' 模块...")
        for key in keys_to_delete:
            logger.debug(f"  删除 sys.modules['{key}']")
            try:
                del sys.modules[key]
            except KeyError:
                logger.debug(f"  键 '{key}' 在尝试删除前已不存在于 sys.modules 中。")
    else:
        logger.info("在 sys.modules 中未找到需要移除的 'transformers_modules.*' 模块。")

    logger.info("--- 动态代码缓存清理完成 ---")

def preload_single_model(model_info):
    """
    尝试预加载单个模型及其相关组件。返回 True/False。
    """
    model_name = model_info["name"]
    logger.info(f"--- === 开始预加载模型: {model_name} === ---")

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    # **关键**: 加载前清理动态模块缓存
    clear_dynamic_modules_cache()

    config, model, tokenizer, processor = None, None, None, None
    success = False

    try:
        # 1. 加载配置
        logger.info(f"[{model_name}] 正在加载配置 (trust_remote_code=True)...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"[{model_name}] 配置加载成功。")

        # 2. 加载模型
        logger.info(f"[{model_name}] 正在加载模型权重 (trust_remote_code=True)...")
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        logger.info(f"[{model_name}] 模型权重加载成功。")

        # 3. 加载 Tokenizer
        if model_info.get("has_tokenizer"):
            logger.info(f"[{model_name}] 正在加载 Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"[{model_name}] Tokenizer 加载成功。")

        # 4. 加载 Processor
        if model_info.get("has_processor"):
            logger.info(f"[{model_name}] 正在加载 Processor...")
            try:
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                logger.info(f"[{model_name}] Processor 加载成功 (使用了 trust_remote_code)。")
            except TypeError:
                logger.warning(f"[{model_name}] Processor 不接受 'trust_remote_code' 参数，尝试不带此参数加载...")
                try:
                    processor = AutoProcessor.from_pretrained(model_name)
                    logger.info(f"[{model_name}] Processor 加载成功 (未使用 trust_remote_code)。")
                except Exception as e_proc_fallback:
                    logger.error(f"[{model_name}] 加载 Processor 失败 (即使未使用 trust_remote_code): {e_proc_fallback}", exc_info=True)
            except Exception as e_proc:
                logger.error(f"[{model_name}] 使用 trust_remote_code 加载 Processor 失败: {e_proc}", exc_info=True)

        logger.info(f"--- === 模型 {model_name} 预加载成功 === ---")
        success = True

    except Exception as e:
        logger.error(f"❌❌❌ 模型 {model_name} 预加载失败: {e} ❌❌❌", exc_info=True)
        success = False

    finally:
        # 清理内存
        logger.debug(f"[{model_name}] 清理内存中的 Python 对象...")
        del config, model, tokenizer, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"[{model_name}] 已清理 CUDA 缓存。")
        logger.debug(f"[{model_name}] 内存清理完成。")

    return success

# --- 主执行逻辑 ---
logger.info(f"===== 开始 Hugging Face 模型预加载流程 =====")
logger.info(f"使用的 HF_HOME (缓存目录): {HF_HOME}") # 仍然记录 HF_HOME，这很重要
logger.info(f"目标模型列表: {[m['name'] for m in MODELS_TO_PRELOAD]}")

overall_success = False
for attempt in range(MAX_ATTEMPTS):
    logger.info(f"🚀 第 {attempt + 1}/{MAX_ATTEMPTS} 次尝试预加载所有模型...")
    models_loaded_this_attempt = 0
    failed_models_this_attempt = []

    for model_info in MODELS_TO_PRELOAD:
        if preload_single_model(model_info):
            models_loaded_this_attempt += 1
        else:
            failed_models_this_attempt.append(model_info["name"])
            # 可选：如果希望一旦失败就立即重试，可以在这里加 break
            # break

    if models_loaded_this_attempt == len(MODELS_TO_PRELOAD):
        logger.info(f"✅✅✅ 第 {attempt + 1} 次尝试成功! 所有模型均已预加载。")
        overall_success = True
        break
    else:
        logger.warning(f"⚠️ 第 {attempt + 1} 次尝试未完全成功。失败的模型: {failed_models_this_attempt}")
        if attempt < MAX_ATTEMPTS - 1:
            wait_time = 5 * (attempt + 1)
            logger.info(f"将在 {wait_time} 秒后进行下一次尝试...")
            time.sleep(wait_time)
        else:
            logger.error("❌❌❌ 已达到最大预加载尝试次数。")

# --- 最终结果 ---
if overall_success:
    logger.info("🎉🎉🎉 所有模型预加载流程成功完成。 🎉🎉🎉")
    sys.exit(0)
else:
    logger.error("🔥🔥🔥 模型预加载流程最终失败。请仔细检查 Docker 构建日志中的错误信息。 🔥🔥🔥")
    sys.exit(1)