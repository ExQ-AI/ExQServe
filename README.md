# ExQServe

**English** | [简体中文](README.zh-CN.md)

Agent-focused OpenAI- and Anthropic-compatible serving for ExLlamaV3 / EXL3.

## Features

- OpenAI and Anthropic compatible APIs, including Chat Completions, Responses, Messages, Completions, Models, and token counting
- Agent workflows with reasoning, tool calling, parallel tool calls, structured output, streaming, cancellation, and continuation
- Agent adaptations for the Qwen3.5 architecture family, Gemma 4, and Muse Glimmer, plus a conservative Generic HF fallback
- Long-context and runtime controls including quantized KV cache, MTP, external draft models, CUDA device selection, and ExLlamaV3 tensor parallelism
- Model switching, PEFT LoRA, YAML configuration, Prometheus metrics, and optional API-key authentication

## Model support

| Model family | Status | Notes |
|---|---|---|
| Qwen3.5 architecture family | Adapted | Covers Qwen3.5 / 3.6 / 3.8; reasoning, tools, parallel tools, and tool-result continuation |
| Gemma 4 family | Adapted | Reasoning, tools, parallel tools, and tool-result continuation |
| Muse Glimmer family | Adapted | ATEM/channel protocol; `low`, `medium`, `high`, and `xhigh` reasoning strengths |
| DeepSeek family | Pending | Agent protocol adaptation is not yet complete |
| GLM family | Pending | Agent protocol adaptation is not yet complete |
| Other compatible Hugging Face models | Generic compatibility | Uses the model's own Hugging Face chat template; reasoning and tool handling remain conservative |

Adapted families preserve their model-native reasoning and tool protocols. Vision has been validated on Qwen3.8, Gemma 4, and Muse Glimmer; Generic HF can preserve multimodal input when the backend exposes a compatible vision component. Image input is opt-in with `--vision`, and unsupported model/backend combinations fail explicitly instead of silently falling back to text mode.

## Installation

ExQServe requires Python 3.12+, an NVIDIA GPU, a CUDA-enabled PyTorch build, and ExLlamaV3 `>=1.4.4,<1.5`.

### Linux

Make sure `python3 --version` is 3.12 or newer. The commands below use PyTorch 2.11.0 with CUDA 12.8 as one known-good runtime combination.

```bash
git clone https://github.com/ExQ-AI/ExQServe.git
cd ExQServe
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Download the `linux_x86_64` ExLlamaV3 wheel that matches your Python version, PyTorch version, and CUDA build from the [ExLlamaV3 v1.4.4 release](https://github.com/turboderp-org/exllamav3/releases/tag/v1.4.4), then install the downloaded wheel and ExQServe:

```bash
pip install ./path/to/exllamav3.whl
pip install .
```

If you use a different Python / PyTorch / CUDA combination, choose the matching wheel from the upstream [ExLlamaV3 releases](https://github.com/turboderp-org/exllamav3/releases).

### Docker

Tagged NVIDIA images are published to `ghcr.io/exq-ai/exqserve`. With a model already available on the host:

```bash
docker pull ghcr.io/exq-ai/exqserve:latest
docker run --rm --gpus all --shm-size 8g \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/exq-ai/exqserve:latest \
  /models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --model-root /models \
  --host 0.0.0.0 --port 8000
```

The NVIDIA driver and NVIDIA Container Toolkit are host requirements. For Docker Compose, copy `docker-compose.env.example` to `.env`, set the model directory and model name, then run:

```bash
docker compose up -d
```

### Windows

Use Python from python.org rather than the Microsoft Store, and make sure `python --version` is 3.12 or newer. The commands below use PyTorch 2.11.0 with CUDA 12.8 as one known-good runtime combination.

```powershell
git clone https://github.com/ExQ-AI/ExQServe.git
cd ExQServe
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -U "triton-windows<3.7"
```

Download the `win_amd64` ExLlamaV3 wheel that matches your Python version (`cp312`, `cp313`, `cp314`, etc.), PyTorch version, and CUDA build from the [ExLlamaV3 v1.4.4 release](https://github.com/turboderp-org/exllamav3/releases/tag/v1.4.4), then install the downloaded wheel and ExQServe:

```powershell
pip install .\path\to\exllamav3.whl
pip install .
```

Using a matching prebuilt ExLlamaV3 wheel avoids a local CUDA-extension build. A plain `pip install exllamav3` uses the PyPI source package and requires local build prerequisites such as Visual Studio Build Tools and a CUDA Toolkit.

## Quick start

For a native Linux or Windows install, verify the CLI first:

```bash
exqserve --help
```

Download the official Qwen3.8-27B EXL3 self-calibrated 4.00 bpw / H5 quant:

```bash
exqserve download turboderp/Qwen3.8-27B-exl3 --revision SC_4.00bpw_H5 --output ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

Start the server:

```bash
exqserve ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 --served-model-id Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

The default address is `http://127.0.0.1:8000`. API-key authentication is optional when binding to localhost.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

On PowerShell, use `curl.exe` for the same commands.

To listen on another interface, configure an API key and add `--host 0.0.0.0 --port 8000`.

## API support

| API | Endpoint |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| OpenAI Responses token counting | `POST /v1/responses/input_tokens` |
| OpenAI Completions | `POST /v1/completions` |
| OpenAI Models | `GET /v1/models`, `GET /v1/models/{model}` |
| Anthropic Messages | `POST /v1/messages` |
| Anthropic token counting | `POST /v1/messages/count_tokens` |
| Mid-stream output injection | `POST /v1/requests/{request_id}/inject` |
| Model management | `GET /admin/models`, `POST /admin/models/load`, `POST /admin/models/switch`, `POST /admin/models/unload` |
| Health | `GET /health` |
| Prometheus metrics | `GET /metrics` |

Output injection accepts a JSON body such as `{"text":"..."}` for an active streaming request. It modifies the current assistant output rather than creating a new user turn; structured-output requests do not support injection.

## Common runtime options

| Option | Description |
|---|---|
| `--served-model-id` | Model name exposed by the API |
| `--model-root` | Directory containing switchable models |
| `--chat-template` | Override the model's HF chat template with a UTF-8 Jinja file |
| `--vision` | Load the model's vision component and accept image input; fails clearly if the selected model/backend cannot provide it |
| `--allow-remote-images` | Allow HTTP(S) image URLs; data-image URLs only require `--vision` |
| `--vision-cache-mb` | CPU budget for cached vision embeddings (default 256 MiB; `0` disables retention) |
| `--max-injection-body-bytes` | Maximum JSON body size for output injection (default 64 KiB) |
| `--cache-tokens` | KV-cache capacity |
| `--kv-cache-bits` | KV-cache precision |
| `--max-batch-size` | Maximum batch size |
| `--max-chunk-size` | Prefill chunk size |
| `--mtp` | Enable MTP speculative decoding |
| `--mtp-draft-tokens` | MTP draft-token count |
| `--mtp-cache-bits` | MTP draft-cache precision |
| `--dynamic-draft` | Enable ExLlamaV3 confidence-calibrated dynamic draft sizing |
| `--draft-confidence` | Target acceptance probability for dynamic draft sizing |
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

CLI arguments override YAML values.

Use `--chat-template PATH` or `chat-template: PATH` in YAML to replace the model's bundled Hugging Face chat template. Dedicated adapters require a template that preserves the model family's reasoning and tool-output protocol.

## More options

```bash
exqserve --help
```

## Acknowledgements

- [ExLlamaV3 / EXL3](https://github.com/turboderp-org/exllamav3)
