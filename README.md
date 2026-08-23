# llama.rs

A Rust shim around [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp) that gives you an Ollama-like CLI and API without the parts that make you want to throw your keyboard out the window. The launcher is tuned for ik_llama.cpp while retaining compatibility with current upstream binaries.

## Why does this exist?

Because Ollama keeps breaking.

Every other update something silently changes — the GGUF loader stops recognizing a format it handled yesterday, a perfectly fine model suddenly fails with a cryptic error about tensor dimensions, or the whole thing just segfaults on a model that worked five minutes ago. You find yourself spelunking through GitHub issues at 2 AM, reading comments like "have you tried rebuilding from source?" and "works on my machine."

Meanwhile, plain `llama.cpp` *just works*. It loads models. It generates text. It doesn't have opinions about your GGUF files. The only problem is that running it raw means typing `llama-server` commands with seventeen flags, and nothing talks to it because everything expects the Ollama API.

So here we are. This project wraps `llama-server` and `llama-cli` in a Rust binary that:

- Spawns and manages the llama.cpp process for you (no more flag salad)
- Exposes an **OpenAI-compatible API** (`/v1/chat/completions`) as a direct passthrough — zero transformation, what goes in comes out
- Fakes an **Ollama-compatible API** (`/api/chat`, `/api/generate`, `/api/tags`) so tools like OpenWebUI and LibreChat think they're talking to Ollama
- Downloads GGUF models from HuggingFace with parallel connections
- Lists and deletes models, stops running servers before removal — basically `ollama` muscle memory but backed by software that doesn't gaslight you

## What this is NOT

- **Not an Ollama replacement.** This covers maybe 30% of what Ollama does. No model registry, no layers, no Modelfiles, no multi-model serving. If Ollama works for you, genuinely, keep using it.
- **Not production software.** No auth, no clustering, no rate limiting. It runs one model on one machine and that's it.
- **Not battle-tested.** This was built over the course of a conversation with LLM, fueled by an mass amount of tokens and spite toward brittle abstractions. There will be bugs.

## Quick start

You need `llama-server` and `llama-cli` from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (recommended) or current upstream llama.cpp installed and in your PATH. Point `LLAMA_BIN_DIR` at the fork's `build/bin` directory if it is not in your PATH.

```bash
# Build
cargo build --release

# Download a model
llama pull mradermacher/Qwen3-8B-GGUF:Q4_K_M

# Interactive REPL
llama run mradermacher/Qwen3-8B-GGUF:Q4_K_M

# Run on CPU only, or pin offload to a particular CUDA GPU
llama run mradermacher/Qwen3-8B-GGUF:Q4_K_M --device cpu
llama run mradermacher/Qwen3-8B-GGUF:Q4_K_M --device gpu1

# Start API server (OpenAI + Ollama compatible)
llama serve mradermacher/Qwen3-8B-GGUF:Q4_K_M

# Or keep the whole experiment in a reusable YAML profile
llama serve --config experiments/qwen3.yaml

# List models
llama ls

# Delete a model
llama rm mradermacher/Qwen3-8B-GGUF:Q4_K_M
```

## Configuration

All configuration via environment variables. No config files. Docker-friendly.

```bash
LLAMA_MODELS_DIR=~/.local/share/llama/models  # Where models are stored
LLAMA_BIN_DIR=/usr/local/bin                  # Where llama-server/llama-cli live (default: PATH)
LLAMA_DEVICE=auto                             # auto, cpu, gpu0, gpu1, CUDA0,...
LLAMA_GPU_LAYERS=999                          # GPU layers to offload (default: all)
LLAMA_CTX_SIZE=32768                          # Context window (default: 32768)
LLAMA_TENSOR_SPLIT=14,12                      # Multi-GPU VRAM split
LLAMA_MAIN_GPU=0                              # Optional primary GPU index
LLAMA_HOST=127.0.0.1                          # Bind address (default: 127.0.0.1)
LLAMA_PORT=8080                               # Port (default: 8080)
LLAMA_FLASH_ATTN=1                            # Flash attention on/off (default: on)
LLAMA_MLOCK=1                                 # Lock model in RAM (default: on)
LLAMA_THREADS=8                               # CPU threads (default: auto-detect)
LLAMA_BATCH_SIZE=2048                         # Batch size (default: 2048)
LLAMA_DOWNLOAD_CONNECTIONS=4                  # Parallel download streams (default: 4)
HF_TOKEN=hf_xxx                              # HuggingFace token for gated models
```

`LLAMA_MODELS_DIR` defaults to your OS data directory (`~/Library/Application Support/llama/models` on macOS, `~/.local/share/llama/models` on Linux).

See [docs/config_reference.md](docs/config_reference.md) for the full list.

### YAML experiment profiles

Both execution commands accept `--config`/`-c`. A profile can contain the model and all execution settings, so no model positional is required:

```bash
llama serve --config examples/ik-llama.yaml
llama run --config examples/ik-llama.yaml

# A positional model or --device overrides the value from YAML
llama serve another-model.gguf --config experiment.yaml --device gpu1
```

See [examples/ik-llama.yaml](examples/ik-llama.yaml) for a complete, commented profile. Unknown YAML keys are rejected to catch misspelled experiment parameters.

## API endpoints

Both API surfaces served on the same port:

| Endpoint | Format | For |
|----------|--------|-----|
| `POST /v1/chat/completions` | SSE (streaming) / JSON | OpenAI clients, LibreChat |
| `GET /v1/models` | JSON | OpenAI clients |
| `POST /api/chat` | NDJSON (streaming) / JSON | Ollama clients, OpenWebUI |
| `POST /api/generate` | NDJSON (streaming) / JSON | Ollama clients |
| `GET /api/tags` | JSON | Ollama clients |
| `POST /api/show` | JSON | Ollama clients |

The OpenAI endpoint is a **raw passthrough** — request and response bytes are forwarded to/from llama.cpp without any parsing or transformation. If llama.cpp supports it, so do we. If llama.cpp returns an error, you get that exact error.

The Ollama endpoints translate between Ollama's NDJSON format and OpenAI's SSE format on the fly.

## How it was built

This entire codebase was written in a single extended conversation with [Claude Opus 4.6](https://claude.ai/code), Anthropic's coding agent. The architecture was planned, the implementation was stepped through methodically (12 implementation steps), and 162 integration tests were written — including tests that run against a real `llama-server` instance and tests that use the actual `ollama` CLI binary to validate API compatibility.

Is it overengineered for a wrapper? Probably. Does it work? Surprisingly, yes.

## License

MIT. Do whatever you want with it.
