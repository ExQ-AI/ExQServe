# ExQServe

**English** | [简体中文](README.zh-CN.md)

Agent-focused OpenAI- and Anthropic-compatible serving for ExLlamaV3 / EXL3.

ExQServe is the serving/API layer; ExLlamaV3 provides the inference backend.

## Why ExQServe?

ExQServe handles Agent-facing serving semantics across model families while preserving model-native reasoning and Tool Calling protocols behind OpenAI- and Anthropic-compatible APIs.

For Tool Calling, schema and boundary handling can be enforced during generation. LLGuidance-backed constrained decoding is combined with Tool Call validation and atomic commit of parallel calls, so malformed or incomplete calls are rejected before they reach the next Agent turn.

Runtime and protocol failures are surfaced with explicit recovery, retryability, and restart states. When recovery is safe, a failed ExLlamaV3 generator is quarantined and rebuilt.

## Features

- OpenAI and Anthropic compatible APIs, including Chat Completions, Responses, Messages, Completions, Models, and token counting
- Agent workflows with reasoning, tool calling, parallel tool calls, OpenAI `strict:true` function tools, LLGuidance-backed constrained decoding, Structured Outputs, streaming, cancellation, and continuation
- Model-native Agent adaptations for the Qwen3.5 architecture family, Gemma 4, Muse Glimmer, DeepSeek V4, and GLM-5, plus a conservative Generic HF fallback
- Pluggable model-dialect API for extending model-native reasoning and Tool Calling protocols
- Generation guarantees with fail-closed Tool Call validation, atomic constrained-parallel batches, protocol-aware output boundaries, and explicit terminal semantics
- Agent-oriented failure and recovery semantics, including context-capacity normalization, protocol-visible recovery facts, and safe ExLlamaV3 generator recovery
- Soft Reasoning Budget handling, automatic output-limit resolution, and an optional Claude Code compatibility profile with model-aware mid-conversation system handling and cache-local prompts
- Long-context and ExLlamaV3 runtime controls including quantized KV cache, system-memory KV/recurrent caches, MTP, n-gram drafting, external draft models, MoE CPU offload/expert splitting, vision offload, CUDA device selection, and tensor parallelism
- Model switching, PEFT LoRA, YAML configuration, Prometheus metrics, and optional API-key authentication

## Model support

| Model family | Status | Notes |
|---|---|---|
| Qwen3.5 architecture family | Adapted | Covers Qwen3.5 / 3.6 / 3.8; reasoning, tools, parallel tools, and tool-result continuation |
| Gemma 4 family | Adapted | Reasoning, tools, parallel tools, and tool-result continuation |
| Muse Glimmer family | Adapted | ATEM/channel protocol; `low`, `medium`, `high`, and `xhigh` reasoning strengths |
| DeepSeek family | Adapted (untested) | Agent protocol adaptation is implemented; GPU validation is still pending |
| GLM family | Adapted (untested) | Agent protocol adaptation is implemented; GPU validation is still pending |
| Other compatible Hugging Face models | Generic compatibility | Uses the model's own Hugging Face chat template; reasoning and tool handling remain conservative |

Adapted families preserve their model-native reasoning and tool protocols. Vision has been validated on Qwen3.8, Gemma 4, and Muse Glimmer; Generic HF can preserve multimodal input when the backend exposes a compatible vision component. Image input is opt-in with `--vision`, and unsupported model/backend combinations fail explicitly instead of silently falling back to text mode.

## Agent workload validation

ExQServe is tested with real Agent clients and long-running Tool Calling workloads in addition to protocol and unit tests.

Release validation covers:

- multi-turn, parallel, named, required, and `strict:true` Tool Calling with tool-result continuation
- constrained generation, Structured Outputs, malformed/incomplete model-output boundaries, and fail-closed Tool Call handling
- long-context continuation, cache-local prompt handling, automatic output budgeting, and soft reasoning budgets
- request cancellation, context-capacity rejection, terminal-state serialization, and protocol-visible failure/recovery facts
- backend generator failure, safe recovery, and restart-required behavior when runtime state cannot be reused safely

Release-specific workload results are published with the corresponding release instead of being kept here as permanent benchmarks.

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
| `--model-dialect` | Select a built-in or installed model Agent dialect; `auto` discovers compatible dialects |
| `--tool-constraint-mode` | Generation-time tool constraints: `off`, `format`, or `schema` |
| `--max-tool-calls-per-generation` | Limit protocol-visible tool calls in one assistant generation |
| `--max-constrained-parallel-tool-calls` | Limit one atomic constrained-parallel tool batch |
| `--anthropic-compatibility-profile` | Optional best-effort Anthropic client profile; use `claude-code` for Claude Code-style workloads |
| `--chat-template` | Override the model's HF chat template with a UTF-8 Jinja file |
| `--vision` | Load the model's vision component and accept image input; fails clearly if the selected model/backend cannot provide it |
| `--vision-offload` | Keep the ExLlamaV3 vision component in pinned host memory to reduce VRAM use |
| `--allow-remote-images` | Allow HTTP(S) image URLs; data-image URLs only require `--vision` |
| `--vision-cache-mb` | CPU budget for cached vision embeddings (default 256 MiB; `0` disables retention) |
| `--max-injection-body-bytes` | Maximum JSON body size for output injection (default 64 KiB) |
| `--cache-tokens` | KV-cache capacity |
| `--kv-cache-bits` | KV-cache precision |
| `--sysmem-kv-cache-mb` | Pinned system-memory budget for ExLlamaV3's second-tier K/V page cache |
| `--sysmem-recurrent-cache-mb` | System-memory budget for ExLlamaV3 recurrent-state checkpoints |
| `--max-in-flight` | Maximum concurrently admitted requests |
| `--max-prompt-tokens` | Optional server-side prompt-token limit |
| `--max-output-tokens` | Optional server-side output-token limit |
| `--max-total-tokens` | Optional server-side prompt + output token limit |
| `--default-output-tokens` | Default API output limit; `auto`/unset lets the serving layer resolve the available output budget |
| `--reasoning-budget-tokens` | Default soft reasoning-token budget; `-1` disables the server default |
| `--reasoning-budget-message` | Optional text inserted inside reasoning immediately before a budget-forced close |
| `--max-batch-size` | Maximum batch size |
| `--max-chunk-size` | Prefill chunk size |
| `--mtp` | Enable MTP speculative decoding |
| `--mtp-draft-tokens` | MTP draft-token count |
| `--mtp-cache-bits` | MTP draft-cache precision |
| `--dynamic-draft` | Enable ExLlamaV3 confidence-calibrated dynamic draft sizing |
| `--draft-confidence` | Target acceptance probability for dynamic draft sizing |
| `--ngram-match-min` | Enable ExLlamaV3 n-gram drafting with the specified minimum history-match length; `0` disables it |
| `--ngram-draft-tokens` | Maximum speculative tokens proposed by n-gram drafting |
| `--draft-model` | External draft model |
| `--moe-cpu-offload-layers` | Run the first N eligible block-sparse MoE layers on CPU |
| `--moe-cpu-split-experts` | Keep N routed experts per eligible MoE layer on CPU through ExLlamaV3 split mode |
| `--draft-moe-cpu-offload-layers` | Run the first N eligible draft/MTP MoE layers on CPU |
| `--moe-cpu-threads` | Worker-thread count for ExLlamaV3 MoE CPU offload |
| `--device-ids` | Process-visible CUDA device allowlist, e.g. `0,1` |
| `--tensor-parallel` | Enable ExLlamaV3 tensor parallelism |
| `--tp-backend` | Select the ExLlamaV3 tensor-parallel communication backend: `native` or `nccl` |
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
