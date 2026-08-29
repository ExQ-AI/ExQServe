# syntax=docker/dockerfile:1

ARG PYTORCH_VERSION=2.11.0
ARG CUDA_VERSION=12.8
ARG CUDNN_VERSION=9

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

ARG EXLLAMAV3_WHEEL_URL="https://github.com/turboderp-org/exllamav3/releases/download/v1.4.4/exllamav3-1.4.4%2Bcu128.torch2.11.0-cp312-cp312-linux_x86_64.whl"

LABEL org.opencontainers.image.source="https://github.com/ExQ-AI/ExQServe" \
      org.opencontainers.image.description="OpenAI- and Anthropic-compatible serving for ExLlamaV3 / EXL3" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "${EXLLAMAV3_WHEEL_URL}"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && python -c "import torch; import exllamav3_ext, fastapi, huggingface_hub, jsonschema, uvicorn; import exllamav3, exqserve; from importlib.metadata import version; assert version('exllamav3').startswith('1.4.4+cu128.torch2.11.0'); assert version('exqserve') == '0.2.1'; assert torch.__version__.startswith('2.11.0')"

EXPOSE 8000

ENTRYPOINT ["exqserve"]
