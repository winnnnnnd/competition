ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

ARG CONDA_ENV_NAME=""
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    CONDA_ENV_NAME=${CONDA_ENV_NAME}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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
RUN if [[ -n "${CONDA_ENV_NAME}" && -f /root/miniconda3/etc/profile.d/conda.sh ]]; then \
      source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV_NAME}"; \
    fi \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app
RUN if [[ -n "${CONDA_ENV_NAME}" && -f /root/miniconda3/etc/profile.d/conda.sh ]]; then \
      source /root/miniconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV_NAME}"; \
    fi \
    && pip install --no-cache-dir --no-deps -e /app \
    && chmod +x /app/scripts/entrypoint.sh /app/scripts/start_ray_worker.sh

EXPOSE 7860 8000 8265 10001
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["streamlit", "run", "apps/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
