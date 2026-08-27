# ExQServe

[English](README.md) | **简体中文**

面向 ExLlamaV3 / EXL3 的 OpenAI 与 Anthropic 兼容推理服务，支持现代 Agent 工作负载。

## 特性

- OpenAI Chat Completions、Responses、Completions、模型发现和 token 计数
- Anthropic Messages 和 token 计数
- 推理、工具调用、结构化输出、流式输出、取消和响应续接
- 量化 KV Cache 与长上下文
- MTP 和外部 draft model
- CUDA 设备选择与 ExLlamaV3 Tensor Parallel 控制
- PEFT LoRA
- YAML 配置、模型切换、Prometheus Metrics 和可选 API Key

## 快速开始

需要 Python 3.12+、NVIDIA GPU、CUDA 12.4+ 和匹配的 PyTorch。请先按照上游 [ExLlamaV3 安装说明](https://github.com/turboderp-org/exllamav3) 安装 ExLlamaV3。

### 1. 克隆并安装

```bash
git clone https://github.com/ExQ-AI/ExQServe.git
cd ExQServe
python -m pip install .
```

### 2. 下载模型

这里直接使用官方 Qwen3.8-27B EXL3 的 SC 4.00 bpw / H5 版本：

```bash
exqserve download turboderp/Qwen3.8-27B-exl3 \
  --revision SC_4.00bpw_H5 \
  --output ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

### 3. 启动服务

```bash
exqserve ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --served-model-id Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

默认地址为 `http://127.0.0.1:8000`，API Key 可选。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

### 4. 发送请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-exl3-SC_4.00bpw_H5",
    "messages": [{"role": "user", "content": "用一句话打个招呼。"}],
    "reasoning_effort": "none",
    "max_completion_tokens": 128
  }'
```

监听其他网卡时可以配置 API Key：

```bash
EXQSERVE_API_KEY='change-me' \
  exqserve ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --served-model-id Qwen3.8-27B-exl3-SC_4.00bpw_H5 \
  --host 0.0.0.0 --port 8000
```

## API 支持

| API | Endpoint |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| OpenAI Responses token 计数 | `POST /v1/responses/input_tokens` |
| OpenAI Completions | `POST /v1/completions` |
| OpenAI 模型发现 | `GET /v1/models`、`GET /v1/models/{model}` |
| Anthropic Messages | `POST /v1/messages` |
| Anthropic token 计数 | `POST /v1/messages/count_tokens` |
| 模型管理 | `GET /admin/models`、`POST /admin/models/load`、`POST /admin/models/switch`、`POST /admin/models/unload` |
| 健康检查 | `GET /health` |
| Prometheus Metrics | `GET /metrics` |

## 常用运行参数

| 参数 | 说明 |
|---|---|
| `--served-model-id` | API 对外显示的模型名称 |
| `--model-root` | 可切换模型所在目录 |
| `--cache-tokens` | KV Cache 容量 |
| `--kv-cache-bits` | KV Cache 精度 |
| `--max-batch-size` | 最大 batch size |
| `--max-chunk-size` | Prefill chunk size |
| `--mtp` | 开启 MTP 投机解码 |
| `--mtp-draft-tokens` | MTP draft token 数量 |
| `--mtp-cache-bits` | MTP draft cache 精度 |
| `--draft-model` | 外部 draft model |
| `--device-ids` | 当前进程可见的 CUDA 设备列表，例如 `0,1` |
| `--tensor-parallel` | 开启 ExLlamaV3 Tensor Parallel |
| `--lora` | 加载 PEFT LoRA |
| `--sampler-preset` | 加载 sampler preset YAML |

一套在 24 GB RTX 4090 上验证过的 Qwen3.8-27B SC 配置：

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

实际可用上下文取决于模型、量化级别、KV Cache 精度、GPU 显存和运行参数。

## YAML 配置

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

CLI 参数优先于 YAML 配置。

## 模型支持

Qwen 模型使用专用适配处理模型特有的 reasoning 和 tool 行为。其他兼容的 Hugging Face 架构可以通过模型自身 chat template 使用 Generic HF 纯文本回退。

## 更多参数

```bash
exqserve --help
```
