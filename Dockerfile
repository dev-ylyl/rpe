FROM runpod/base:0.6.2-cuda11.8.0

ENV HF_HOME=/runpod-volume

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y ffmpeg git
RUN python3 -m pip install --upgrade pip &&     python3 -m pip install -r requirements.txt --no-cache-dir

RUN pip uninstall torch -y &&     pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

CMD ["python3", "-u", "main.py"]
