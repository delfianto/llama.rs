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
| `LLAMA_GPU_LAYERS` | `999` | Number of layers to offload to GPU (999 = all) |
| `LLAMA_TENSOR_SPLIT` | *(none)* | VRAM ratio per GPU, comma-separated (e.g., `14,12`) |
| `LLAMA_MAIN_GPU` | `0` | Primary GPU device index |
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

### REPL (`llama run` only)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SYSTEM_PROMPT` | `You are a helpful assistant.` | System prompt for interactive mode |

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
llama run <model>                      Start interactive REPL
llama serve <model>                    Start API server
llama pull <org/repo> <filename>       Download GGUF from HuggingFace
llama ls                               List downloaded models
llama rm <model>                       Delete a model
llama --help                           Show help
llama --version                        Show version
```

### Model Argument Resolution

The `<model>` argument is resolved in order:
1. If absolute path → use as-is
2. If contains `/` → treat as relative path under `$LLAMA_MODELS_DIR`
3. Otherwise → search `$LLAMA_MODELS_DIR` recursively for a matching filename

### Examples

```bash
# Direct filename (searched in models dir)
llama run qwen3-14b-q4_k_m.gguf

# Relative path under models dir
llama serve mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF/model.gguf

# Absolute path
llama run /mnt/nvme/models/phi-4.gguf

# Download a model
llama pull mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF Q4_K_M.gguf

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
