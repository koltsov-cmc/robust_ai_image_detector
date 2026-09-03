# docker/Dockerfile
# GPU-ready base image with PyTorch + CUDA
# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

COPY . /workspace

ENTRYPOINT ["python", "/workspace/exp_random.py"]
