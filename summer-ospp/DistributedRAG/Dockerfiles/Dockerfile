ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    fonts-noto-cjk \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libreoffice-calc \
    libreoffice-impress \
    libreoffice-writer \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app
RUN chmod +x /app/Dockerfiles/entrypoint.sh /app/Dockerfiles/start-ray-worker.sh

EXPOSE 7860 8000 8265 10001
ENTRYPOINT ["/app/Dockerfiles/entrypoint.sh"]
CMD ["streamlit", "run", "main_app2/main.py", "--server.port=7860", "--server.address=0.0.0.0"]
