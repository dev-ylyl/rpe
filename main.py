import runpod
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def handler(job):
    # Start handling the job
    print(f"Worker Start")
    
    model_type = job["input"].get("model", "text")
    inputs = job["input"].get("data")
    
    logging.info(f"📋 inputs内容是: {inputs}")
    logging.info(f"📋 model_type内容是: {model_type}")

    if isinstance(inputs, str):
        inputs = [inputs]

    if not inputs:
        logging.warning("⚠️ 数据为空")
        return {
            "output": {
                "error": "Empty input provided."
            }
        }
    
    logging.info(f"✅ 处理完成，输入数量: {len(inputs)}")
    return {
        "output": {
            "message": f"Received {len(inputs)} inputs.",
            "inputs": inputs,
            "model_type": model_type
        }
    }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})