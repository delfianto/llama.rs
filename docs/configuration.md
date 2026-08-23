# Configuration

`llama.rs` supports environment variables for machine-wide defaults and YAML
profiles for repeatable model experiments.

## Precedence

Values are applied in this order, from lowest to highest priority:

1. Built-in defaults
2. Environment variables
3. YAML profile passed with `--config` or `-c`
4. Positional `MODEL` and CLI `--device`

Only `run` and `serve` load YAML profiles. A model may be supplied by the profile,
the positional argument, or both.

```bash
llama serve --config profiles/model.yaml
llama serve different.gguf --config profiles/model.yaml --device gpu1
```

## Environment variables

### Paths

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_BIN_DIR` | Search `PATH` | Directory containing `llama-server` and `llama-cli` |
| `LLAMA_MODELS_DIR` | OS data directory | GGUF root; normally `~/.local/share/llama/models` on Linux and `~/Library/Application Support/llama/models` on macOS |

### Compute

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_DEVICE` | `auto` | `auto`, `cpu`, `gpuN`, a native backend device such as `CUDA0`/`Vulkan0`, or a comma-separated list |
| `LLAMA_GPU_LAYERS` | `999` | Layers to offload; `999` conventionally means all available layers |
| `LLAMA_TENSOR_SPLIT` | Engine default | Comma-separated multi-GPU proportions, for example `14,9` |
| `LLAMA_MAIN_GPU` | Engine default | Primary GPU index |
| `LLAMA_FLASH_ATTN` | `1` | Flash attention: `1`/`true`/`yes` to enable, `0` to disable |
| `LLAMA_MLOCK` | `1` | Lock model memory to avoid swapping |

`gpuN` is normalized to ik_llama.cpp's `CUDAN` device name. CPU mode sets GPU
layers to zero and ignores device placement, main GPU, and tensor split settings.

### Inference

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_CTX_SIZE` | `32768` | Context window in tokens |
| `LLAMA_BATCH_SIZE` | `2048` | Prompt-processing batch size |
| `LLAMA_THREADS` | Available CPU count | CPU computation threads |

### Server

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_HOST` | `127.0.0.1` | Public proxy bind address |
| `LLAMA_PORT` | `8080` | Public proxy port |

### Prompt and sampling

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful assistant.` | Interactive-mode system prompt |
| `LLAMA_SYSTEM_PROMPT_FILE` | Unset | System prompt file; takes priority over the string value when readable |
| `LLAMA_PROMPT_TEMPLATE` | Unset | Chat template string passed to llama.cpp |
| `LLAMA_PROMPT_TEMPLATE_FILE` | Unset | Chat template file; takes priority over the string value when it exists |
| `LLAMA_TEMPERATURE` | Engine default | Sampling temperature |
| `LLAMA_MAX_TOKENS` | Engine default | Maximum generated tokens |
| `LLAMA_CTX_OVERFLOW` | `shift` | `shift` or `stop`; `stop` emits `--no-context-shift` |
| `LLAMA_STOP` | Unset | Comma-separated reverse prompts for `llama run` only |
| `LLAMA_TOP_K` | Engine default | Top-k sampling |
| `LLAMA_TOP_P` | Engine default | Nucleus sampling probability |
| `LLAMA_MIN_P` | Engine default | Minimum probability sampling |
| `LLAMA_REPEAT_PENALTY` | Engine default | Repetition penalty |
| `LLAMA_PRESENCE_PENALTY` | Engine default | Presence penalty |

Unset sampling values are left to the selected llama.cpp build. API clients may
override server sampling values per request.

### Downloads and logging

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_DOWNLOAD_CONNECTIONS` | `4` | Parallel model download connections |
| `HF_TOKEN` | Unset | Hugging Face token for gated repositories |
| `LLAMA_LOG` | `info` | Wrapper log level |
| `RUST_LOG` | Unset | Fine-grained tracing filter; takes priority over `LLAMA_LOG` |
| `LLAMA_LOG_VERBOSITY` | `0` | llama.cpp child-process verbosity |

Engine stdout and stderr are always inherited by the terminal. The verbosity
variable changes what llama.cpp emits; it does not control whether the wrapper
shows that output.

## YAML profiles

Profiles reject unknown keys, which catches misspelled parameters early. This is
the complete schema:

```yaml
model: /models/Qwen3-30B-A3B-IQ4_XS.gguf

paths:
  bin_dir: /opt/ik_llama.cpp/build/bin
  models_dir: /models

compute:
  device: gpu0,gpu1
  gpu_layers: 999
  tensor_split: 14,9
  main_gpu: 0
  flash_attention: true
  mlock: true

inference:
  context_size: 16384
  batch_size: 1024
  threads: 16

server:
  host: 127.0.0.1
  port: 8080

prompt:
  system: You are a concise assistant.
  # system_file: prompts/system.txt
  # chat_template: chatml
  # chat_template_file: templates/model.jinja
  stop:
    - "<|end|>"

sampling:
  temperature: 0.6
  max_tokens: 4096
  context_overflow: shift
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
```

All fields are optional. Relative `bin_dir`, `models_dir`, `system_file`, and
`chat_template_file` paths resolve relative to the profile file, not the current
working directory. File values take priority over inline prompt values.

The repository includes a [complete commented example](../examples/ik-llama.yaml).

## Passing engine-specific arguments

`extra_args` is an ordered list passed directly to both `llama-cli` and
`llama-server`, without shell parsing:

```yaml
extra_args:
  - --draft-model
  - /models/draft.gguf
  - --draft-max
  - "8"
```

Keep each argument and value as a separate YAML item. Quote values when YAML could
interpret their type. The wrapper appends its private server `--host` and `--port`
after `extra_args`, so those transport settings cannot be replaced through this
list; use the `server` section for the public bind address.

Engine-specific flags vary by llama.cpp fork and build. Confirm them with:

```bash
llama-server --help
llama-cli --help
```

## Common tuning examples

CPU-only:

```yaml
compute:
  device: cpu
```

Two CUDA devices with an explicit split:

```yaml
compute:
  device: gpu0,gpu1
  tensor_split: 14,9
  main_gpu: 0
```

Reduce memory usage:

```yaml
compute:
  gpu_layers: 40
inference:
  context_size: 8192
  batch_size: 512
```

Memory estimates printed while loading should include the main model, KV cache,
compute buffers, and any draft/MTP model configured through engine arguments.
