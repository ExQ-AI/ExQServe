# ExQServe

**English** | [简体中文](README.zh-CN.md)

OpenAI- and Anthropic-compatible serving for ExLlamaV3 / EXL3, with support for modern Agent workloads.

## Features

- OpenAI Chat Completions, Responses, Completions, model discovery, and token counting
- Anthropic Messages and token counting
- Reasoning, tool calling, structured output, streaming, cancellation, and response continuation
- Quantized KV cache and long-context serving
- MTP and external draft models
- CUDA device selection and ExLlamaV3 tensor-parallel controls
- PEFT LoRA support
- YAML configuration, model switching, Prometheus metrics, and optional API-key authentication

## Quick start

Requires Python 3.12+, an NVIDIA GPU, CUDA 12.4+, and a matching PyTorch build. Install ExLlamaV3 first using the upstream [installation instructions](https://github.com/turboderp-org/exllamav3).

### 1. Clone and install

```bash
git clone https://github.com/ExQ-AI/ExQServe.git
cd ExQServe
python -m pip install .
```

### 2. Download a model

This example uses the official Qwen3.8-27B EXL3 self-calibrated 4.00 bpw / H5 quant:

```bash
exqserve download turboderp/Qwen3.8-27B-exl3 \
  --revision SC_4.00bpw_H5 \
  --output ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

### 3. Start the server

```bash
exqserve ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --served-model-id Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

The default address is `http://127.0.0.1:8000` and API-key authentication is optional.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

### 4. Send a request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-exl3-SC_4.00bpw_H5",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "reasoning_effort": "none",
    "max_completion_tokens": 128
  }'
```

To listen on another interface, configure an API key:

```bash
EXQSERVE_API_KEY='change-me' \
  exqserve ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --served-model-id Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --host 0.0.0.0 --port 8000
```

## API support

| API | Endpoint |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| OpenAI Responses token counting | `POST /v1/responses/input_tokens` |
| OpenAI Completions | `POST /v1/completions` |
| OpenAI model discovery | `GET /v1/models`, `GET /v1/models/{model}` |
| Anthropic Messages | `POST /v1/messages` |
| Anthropic token counting | `POST /v1/messages/count_tokens` |
| Model management | `GET /admin/models`, `POST /admin/models/load`, `POST /admin/models/switch`, `POST /admin/models/unload` |
| Health | `GET /health` |
| Prometheus metrics | `GET /metrics` |

## Common runtime options

| Option | Description |
|---|---|
| `--served-model-id` | Model name exposed by the API |
| `--model-root` | Directory containing switchable models |
| `--cache-tokens` | KV-cache capacity |
| `--kv-cache-bits` | KV-cache precision |
| `--max-batch-size` | Maximum batch size |
| `--max-chunk-size` | Prefill chunk size |
| `--mtp` | Enable MTP speculative decoding |
| `--mtp-draft-tokens` | MTP draft-token count |
| `--mtp-cache-bits` | MTP draft-cache precision |
| `--draft-model` | External draft model |
| `--device-ids` | Process-visible CUDA device allowlist, e.g. `0,1` |
| `--tensor-parallel` | Enable ExLlamaV3 tensor parallelism |
| `--lora` | Load a PEFT LoRA adapter |
| `--sampler-preset` | Load a sampler preset YAML |

A Qwen3.8-27B SC profile validated on a 24 GB RTX 4090:

```bash
exqserve ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --served-model-id Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --cache-tokens 262144 \
  --kv-cache-bits 8 \
  --max-batch-size 1 \
  --max-chunk-size 2048 \
  --reserve-per-device-gb 0.09375 \
  --mtp \
  --mtp-draft-tokens 4 \
  --mtp-cache-bits 4 \
  --autosplit-no-forward \
  --cuda-malloc-async \
  --qc-staging 0
```

Available context depends on the model, quantization, KV-cache precision, GPU memory, and runtime settings.

## YAML configuration

```yaml
model-directory: ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5
served-model-id: Qwen3.8-27B-exl3-SC_4.00bpw_H5
host: 127.0.0.1
port: 8000
cache-tokens: 32768
kv-cache-bits: 8
max-chunk-size: 2048
mtp: true
mtp-draft-tokens: 4
mtp-cache-bits: 4
```

```bash
exqserve --config config.yaml
```

CLI arguments override values from the YAML file.

## Model support

Qwen models use dedicated handling for model-specific reasoning and tool behavior. Other compatible Hugging Face architectures can use the Generic HF text fallback based on the model's own chat template.

## More options

```bash
exqserve --help
```
