FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip install --default-timeout=1000 --no-cache-dir "torch==2.3.0+cpu" --index-url https://download.pytorch.org/whl/cpu  

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY src/ src/

COPY data/index/ data/index/
COPY outputs/final_model/ outputs/final_model/

ENTRYPOINT ["python", "-m", "src.predict"]
