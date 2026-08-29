# ExQServe

[English](README.md) | **简体中文**

ExQServe 是面向 ExLlamaV3 / EXL3 的推理服务，兼容 OpenAI 和 Anthropic API，重点面向 Agent 场景。

## 主要功能

- 同时兼容 OpenAI 和 Anthropic API，包括 Chat Completions、Responses、Messages、Completions、Models 和 token 计数
- 面向 Agent 场景支持思考内容与最终回答分离、工具调用、并行工具调用、结构化输出、流式响应、请求取消和连续调用
- 已适配 Qwen3.5 架构系列、Gemma 4、Muse Glimmer 等模型系列；其他兼容 Hugging Face 模型可走通用兼容路径
- 支持长上下文、量化 KV Cache、MTP、外部 draft model、CUDA 设备选择和 ExLlamaV3 Tensor Parallel
- 支持模型切换、PEFT LoRA、YAML 配置、Prometheus Metrics 和可选 API Key

## 模型支持

| 模型系列 | 状态 | 说明 |
|---|---|---|
| Qwen3.5 架构系列 | 已适配 | 覆盖 Qwen3.5 / 3.6 / 3.8；支持思考、工具调用、并行工具调用，工具结果回传后可继续对话 |
| Gemma 4 系列 | 已适配 | 思考、工具调用、并行工具调用，工具结果回传后可继续对话 |
| Muse Glimmer 系列 | 已适配 | ATEM/channel 协议；支持 `low`、`medium`、`high`、`xhigh` 四档思考强度 |
| DeepSeek 系列 | 已适配（未测试） | Agent 协议适配已完成，尚未进行 GPU 实测 |
| GLM 系列 | 已适配（未测试） | Agent 协议适配已完成，尚未进行 GPU 实测 |
| 其他兼容 Hugging Face 模型 | 通用兼容 | 使用模型自带的 Hugging Face chat template；思考和工具调用按保守方式处理 |

已适配系列会保留各自的思考与工具调用格式。Qwen3.8、Gemma 4 和 Muse Glimmer 已验证图片输入；其他 Hugging Face 模型在后端提供兼容视觉组件时也可以保留多模态输入。图片能力需要显式开启 `--vision`，不支持的模型或后端会直接报错，不会静默退回纯文本模式。

## 开发计划

ExQServe 仍在持续开发中，目前主要关注：

- [ ] Dialect 插件系统，降低模型原生 Agent 协议的扩展与维护成本
- [ ] 基于 LLGuidance 的 Constrained Decoding，用于 Tool Calling 与 Structured Outputs
- [ ] 更多模型系列适配

## 安装

ExQServe 需要 Python 3.12+、NVIDIA GPU、支持 CUDA 的 PyTorch，以及 ExLlamaV3 `>=1.4.4,<1.5`。

### Linux

先确认 `python3 --version` 为 3.12 或更高。下面以 PyTorch 2.11.0 + CUDA 12.8 作为一套已验证的运行时组合：

```bash
git clone https://github.com/ExQ-AI/ExQServe.git
cd ExQServe
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

从 [ExLlamaV3 v1.4.4 Release](https://github.com/turboderp-org/exllamav3/releases/tag/v1.4.4) 下载与当前 Python、PyTorch、CUDA 对应的 `linux_x86_64` wheel，然后安装 ExLlamaV3 和 ExQServe：

```bash
pip install ./path/to/exllamav3.whl
pip install .
```

如果使用其他 Python / PyTorch / CUDA 组合，请到上游 [ExLlamaV3 Releases](https://github.com/turboderp-org/exllamav3/releases) 选择匹配的 wheel。

### Docker

NVIDIA 镜像发布在 `ghcr.io/exq-ai/exqserve`。如果模型已经在宿主机上，可以直接启动：

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

宿主机需要安装 NVIDIA 驱动和 NVIDIA Container Toolkit。使用 Docker Compose 时，将 `docker-compose.env.example` 复制为 `.env`，填好模型目录和模型名称后运行：

```bash
docker compose up -d
```

### Windows

建议使用 python.org 提供的 Python，不要使用 Microsoft Store 版本，并确认 `python --version` 为 3.12 或更高。下面仍以 PyTorch 2.11.0 + CUDA 12.8 作为已验证组合：

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

从 [ExLlamaV3 v1.4.4 Release](https://github.com/turboderp-org/exllamav3/releases/tag/v1.4.4) 下载与当前 Python、PyTorch、CUDA 对应的 `win_amd64` wheel。Python 版本对应 wheel 文件名里的 `cp312`、`cp313`、`cp314` 等标签。下载后安装 ExLlamaV3 和 ExQServe：

```powershell
pip install .\path\to\exllamav3.whl
pip install .
```

使用匹配的预编译 wheel 时，不需要在本机编译 ExLlamaV3 的 CUDA 扩展。直接执行 `pip install exllamav3` 会走 PyPI 源码包，需要额外准备 Visual Studio Build Tools 和 CUDA Toolkit 等编译环境。

## 快速开始

Linux 或 Windows 原生安装完成后，先确认命令可用：

```bash
exqserve --help
```

下载官方 Qwen3.8-27B EXL3 SC 4.00 bpw / H5 模型：

```bash
exqserve download turboderp/Qwen3.8-27B-exl3 --revision SC_4.00bpw_H5 --output ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

启动服务：

```bash
exqserve ./models/Qwen3.8-27B-exl3-SC_4.00bpw_H5 --served-model-id Qwen3.8-27B-exl3-SC_4.00bpw_H5
```

默认监听 `http://127.0.0.1:8000`。只监听本机时可以不设置 API Key。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

PowerShell 下可将 `curl` 换成 `curl.exe`。

如果需要监听其他网卡，请设置 API Key，并在启动参数中加入 `--host 0.0.0.0 --port 8000`。

## API 支持

| API | Endpoint |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| OpenAI Responses token 计数 | `POST /v1/responses/input_tokens` |
| OpenAI Completions | `POST /v1/completions` |
| OpenAI 模型列表 / 查询 | `GET /v1/models`、`GET /v1/models/{model}` |
| Anthropic Messages | `POST /v1/messages` |
| Anthropic token 计数 | `POST /v1/messages/count_tokens` |
| 生成中注入文本 | `POST /v1/requests/{request_id}/inject` |
| 模型管理 | `GET /admin/models`、`POST /admin/models/load`、`POST /admin/models/switch`、`POST /admin/models/unload` |
| 健康检查 | `GET /health` |
| Prometheus Metrics | `GET /metrics` |

生成中注入接口接收 `{"text":"..."}` 这样的 JSON body，用于仍在进行的流式请求。它修改的是当前这次模型输出，不会新增用户消息；结构化输出请求不支持这一能力。

## 常用参数

| 参数 | 说明 |
|---|---|
| `--served-model-id` | API 对外显示的模型名称 |
| `--model-root` | 可切换模型所在目录 |
| `--chat-template` | 使用 UTF-8 Jinja 文件覆盖模型自带的 HF chat template |
| `--vision` | 加载模型的视觉组件并接受图片输入；模型或后端不支持时会直接报错 |
| `--allow-remote-images` | 允许 HTTP(S) 图片地址；data URL 只需要开启 `--vision` |
| `--vision-cache-mb` | 图片 embedding 的 CPU 缓存上限，默认 256 MiB；设为 `0` 可关闭缓存 |
| `--max-injection-body-bytes` | 生成中注入接口的 JSON body 上限，默认 64 KiB |
| `--cache-tokens` | KV Cache 容量 |
| `--kv-cache-bits` | KV Cache 精度 |
| `--max-batch-size` | 最大 batch size |
| `--max-chunk-size` | Prefill chunk size |
| `--mtp` | 开启 MTP 投机解码 |
| `--mtp-draft-tokens` | MTP draft token 数量 |
| `--mtp-cache-bits` | MTP draft cache 精度 |
| `--dynamic-draft` | 开启 ExLlamaV3 动态 draft 长度 |
| `--draft-confidence` | Dynamic Draft 的目标接受概率 |
| `--draft-model` | 外部 draft model |
| `--device-ids` | 当前进程可见的 CUDA 设备，例如 `0,1` |
| `--tensor-parallel` | 开启 ExLlamaV3 Tensor Parallel |
| `--lora` | 加载 PEFT LoRA |
| `--sampler-preset` | 加载 sampler preset YAML |

下面是一套已在 24 GB RTX 4090 上验证过的 Qwen3.8-27B SC 配置：

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

CLI 参数优先于 YAML。

如需替换模型自带的 Hugging Face chat template，可使用 `--chat-template PATH`，或在 YAML 中设置 `chat-template: PATH`。对于已经做过专门适配的模型，自定义模板需要保留该模型族原有的思考和工具输出格式。

## 更多参数

```bash
exqserve --help
```

## 致谢

- [ExLlamaV3 / EXL3](https://github.com/turboderp-org/exllamav3)
