# Configuration Reference

## Environment Variables

All configuration is via environment variables, matching the original `llama.sh` script. No config files.

### Path Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_BIN_DIR` | (search `$PATH`) | Directory containing `llama-server` and `llama-cli` binaries |
| `LLAMA_MODELS_DIR` | `~/.local/share/llama/models` | Root directory for GGUF model files |

### GPU / Hardware

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_DEVICE` | `auto` | Compute resource. Accepts `cpu`, `gpu0`, `gpu1`, native backend names such as `CUDA0` or `Vulkan0`, and comma-separated device lists. `gpuN` is shorthand for ik_llama.cpp's `CUDAN`. |
| `LLAMA_GPU_LAYERS` | `999` | Number of layers to offload to GPU (999 = all) |
| `LLAMA_TENSOR_SPLIT` | *(none)* | VRAM ratio per GPU, comma-separated (e.g., `14,12`) |
| `LLAMA_MAIN_GPU` | *(llama.cpp default)* | Optional primary GPU device index. Only emitted when explicitly set. |
| `LLAMA_FLASH_ATTN` | `1` | Flash attention: `1` = on, `0` = off |
| `LLAMA_MLOCK` | `1` | Lock model in RAM to prevent swap: `1` = on, `0` = off |

### Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CTX_SIZE` | `32768` | Context window size in tokens |
| `LLAMA_BATCH_SIZE` | `2048` | Batch size for prompt processing |
| `LLAMA_THREADS` | *(num CPUs)* | CPU threads for computation |

### Server (`llama serve` only)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_HOST` | `127.0.0.1` | Bind address for the proxy server |
| `LLAMA_PORT` | `8080` | Port for the proxy server |

### REPL / Prompt

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt for interactive mode |
| `LLAMA_SYSTEM_PROMPT_FILE` | *(none)* | Path to file containing system prompt. Overrides `LLAMA_SYSTEM_PROMPT` when set and readable. If file is missing/unreadable, warns and falls back. |
| `LLAMA_PROMPT_TEMPLATE_FILE` | *(none)* | Path to a Jinja2 chat template file. Passed as `--chat-template-file` to llama.cpp. Overrides `LLAMA_PROMPT_TEMPLATE` when set and file exists. |
| `LLAMA_PROMPT_TEMPLATE` | *(none)* | Chat template string. Passed as `--chat-template` to llama.cpp. Overridden by `LLAMA_PROMPT_TEMPLATE_FILE`. |

### Sampling Defaults

These set default sampling parameters for both `llama serve` and `llama run`. When unset, llama.cpp's own defaults are used. For `llama serve`, clients can override these per-request via the API.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_TEMPERATURE` | *(llama.cpp default)* | Sampling temperature (e.g., `0.7`) |
| `LLAMA_MAX_TOKENS` | *(llama.cpp default)* | Maximum response tokens (e.g., `2048`) |
| `LLAMA_CTX_OVERFLOW` | `shift` | Context overflow behavior: `shift` (shift context window) or `stop` (stop generating). Maps to `--no-context-shift` when `stop`. |
| `LLAMA_STOP` | *(none)* | Stop strings, comma-separated (e.g., `<\|end\|>,###`). **`llama run` only** — passed as `-r` (reverse prompt) flags. |
| `LLAMA_TOP_K` | *(llama.cpp default)* | Top-k sampling (e.g., `40`) |
| `LLAMA_REPEAT_PENALTY` | *(llama.cpp default)* | Repeat penalty (e.g., `1.1`) |
| `LLAMA_PRESENCE_PENALTY` | *(llama.cpp default)* | Presence penalty (e.g., `0.0`) |
| `LLAMA_TOP_P` | *(llama.cpp default)* | Top-p / nucleus sampling (e.g., `0.9`) |
| `LLAMA_MIN_P` | *(llama.cpp default)* | Min-p sampling (e.g., `0.05`) |

### Download

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_DOWNLOAD_CONNECTIONS` | `4` | Parallel connections for model downloads |
| `HF_TOKEN` | *(none)* | HuggingFace token for gated models |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_LOG` | `info` | Log level: `trace`, `debug`, `info`, `warn`, `error` |
| `RUST_LOG` | *(none)* | Fine-grained tracing filter (overrides `LLAMA_LOG` if set) |

## CLI Commands

```
llama run [<model>] [-c FILE] [--device DEVICE]    Start interactive REPL
llama serve [<model>] [-c FILE] [--device DEVICE]  Start API server
llama pull <org/repo:quant>            Download GGUF from HuggingFace
llama ls                               List downloaded models
llama rm <model>                       Delete a model
llama --help                           Show help
llama --version                        Show version
```

### YAML Execution Profiles

`llama run` and `llama serve` accept `--config FILE` (or `-c FILE`). The profile is overlaid on environment-derived configuration. A positional model and `--device` take final precedence, which makes it easy to reuse a profile while changing only the model or GPU.

```yaml
model: /models/Qwen3-30B-A3B-IQ4_XS.gguf

paths:
  bin_dir: /opt/ik_llama.cpp/build/bin
  models_dir: /models

compute:
  device: gpu1                 # cpu, gpu0, gpu1, CUDA0, Vulkan0, ...
  gpu_layers: 999
  tensor_split: 1,1
  main_gpu: 0
  flash_attention: true
  mlock: true

inference:
  context_size: 65536
  batch_size: 4096
  threads: 12

server:
  host: 127.0.0.1
  port: 8080

prompt:
  system: You are a concise assistant.
  # system_file: prompts/system.txt
  # chat_template: chatml
  # chat_template_file: templates/model.jinja
  stop: ["<|end|>"]

sampling:
  temperature: 0.6
  max_tokens: 4096
  context_overflow: shift       # shift or stop
  top_k: 20
  top_p: 0.95
  min_p: 0.05
  repeat_penalty: 1.1
  presence_penalty: 0.0

extra_args:
  - --cache-type-k
  - q8_0
  - --cache-type-v
  - q8_0
  - --split-mode
  - graph
```

Relative paths in `paths`, `system_file`, and `chat_template_file` resolve from the YAML file's directory. `extra_args` is passed directly as individual process arguments without shell evaluation. Generated internal server `--host` and `--port` arguments are appended afterward and cannot be replaced through `extra_args`.

The complete commented example is in [`examples/ik-llama.yaml`](../examples/ik-llama.yaml).

### Model Directory Structure

Models are stored in LM Studio-compatible 3-level structure:

```
$LLAMA_MODELS_DIR/
└── org/
    └── repo/
        └── model-file.gguf
```

This allows models downloaded by llama.rs to be detected by LM Studio and vice versa.

### Model Argument Resolution

The `<model>` argument is resolved in order:
1. If absolute path → use as-is
2. If `org/repo:quant` spec → search `$LLAMA_MODELS_DIR/org/repo/` for a `.gguf` file matching the quant tag
3. If contains `/` → treat as relative path under `$LLAMA_MODELS_DIR`
4. Otherwise → search `$LLAMA_MODELS_DIR` recursively for a matching filename

### Examples

```bash
# Direct filename (searched in models dir)
llama run qwen3-14b-q4_k_m.gguf

# Relative path under models dir
llama serve mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF/model.gguf

# Absolute path
llama run /mnt/nvme/models/phi-4.gguf

# CPU-only (forces --n-gpu-layers 0 and ignores GPU placement settings)
llama run qwen3-14b-q4_k_m.gguf --device cpu

# Pin ik_llama.cpp offload to the second CUDA GPU (--device CUDA1)
llama serve qwen3-14b-q4_k_m.gguf --device gpu1

# Use selected devices for multi-GPU offload
LLAMA_TENSOR_SPLIT=1,1 llama serve qwen3-14b-q4_k_m.gguf --device gpu0,gpu2

# Download a model
llama pull mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF:Q4_K_M

# Docker usage
docker run -e LLAMA_GPU_LAYERS=40 -e LLAMA_CTX_SIZE=8192 \
  -v /models:/models -e LLAMA_MODELS_DIR=/models \
  llama-rs serve model.gguf
```

## Port Mapping

The proxy server exposes both API surfaces on the same port:

| Endpoint | Protocol | Compatibility |
|----------|----------|---------------|
| `/v1/chat/completions` | SSE | OpenAI / llama.cpp clients |
| `/v1/models` | JSON | OpenAI / llama.cpp clients |
| `/api/chat` | NDJSON | Ollama clients (OpenWebUI, LibreChat) |
| `/api/generate` | NDJSON | Ollama clients |
| `/api/tags` | JSON | Ollama clients |
| `/api/show` | JSON | Ollama clients |
