# Architecture

## Overview

llama.rs is a Rust CLI that wraps llama.cpp binaries in an Ollama-like interface. It spawns `llama-server` or `llama-cli` as child processes, manages model files, and exposes both OpenAI-compatible and Ollama-compatible HTTP APIs.

## Directory / Module Structure

```
src/
├── main.rs              # Entry point, clap CLI definition
├── cli/
│   ├── mod.rs           # CLI command dispatch
│   ├── run.rs           # `llama run <model>` — interactive REPL
│   ├── serve.rs         # `llama serve <model>` — start API server
│   ├── pull.rs          # `llama pull <org/repo> <file>` — download model
│   ├── ls.rs            # `llama ls` — list models
│   └── rm.rs            # `llama rm <model>` — delete model
├── config/
│   ├── mod.rs           # Config struct, env var loading, defaults
│   └── resolve.rs       # Model path resolution (relative → absolute)
├── process/
│   ├── mod.rs           # ProcessManager trait
│   ├── server.rs        # Spawn & manage llama-server subprocess
│   ├── cli.rs           # Spawn & manage llama-cli subprocess (REPL)
│   └── health.rs        # Health check / readiness polling for llama-server
├── api/
│   ├── mod.rs           # Axum router construction
│   ├── openai.rs        # /v1/chat/completions, /v1/models — passthrough proxy
│   ├── ollama.rs        # /api/chat, /api/generate, /api/tags — Ollama translation
│   ├── stream/
│   │   ├── mod.rs       # Shared streaming infrastructure
│   │   ├── sse.rs       # SSE formatter (OpenAI format)
│   │   └── ndjson.rs    # NDJSON formatter (Ollama format)
│   └── types.rs         # Shared request/response types
├── download/
│   ├── mod.rs           # Download manager
│   ├── hf.rs            # HuggingFace API client (file URL resolution)
│   └── progress.rs      # Progress bar / reporting
├── model/
│   ├── mod.rs           # ModelManager — scan, list, delete
│   └── types.rs         # ModelInfo struct
└── error.rs             # Error types
```

## Component Design

### 1. CLI Layer (`cli/`)

Uses `clap` derive macros. Top-level commands:

```
llama run <model>       Start interactive REPL via llama-cli
llama serve <model>     Start HTTP server via llama-server
llama pull <spec>       Download GGUF from HuggingFace
llama ls                List downloaded models
llama rm <model>        Delete a model (stops running process first)
```

`<model>` accepts:
- A filename (resolved against `LLAMA_MODELS_DIR`): `qwen3-14b-q4_k_m.gguf`
- A relative path under models dir: `mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF/model.gguf`
- An absolute path: `/mnt/models/qwen3.gguf`

### 2. Config Layer (`config/`)

All configuration via environment variables with sensible defaults, matching the original shell script. No config files — env vars are Docker-native.

See `CONFIG_REFERENCE.md` for the full list.

Config is loaded once at startup into an immutable `Config` struct, passed by `Arc<Config>` to all components.

### 3. Process Manager (`process/`)

Core responsibility: spawn llama.cpp binaries as child processes and manage their lifecycle.

**For `llama serve`:**
1. Spawn `llama-server` with flags built from Config (GPU layers, tensor split, context size, etc.)
2. Poll `/health` endpoint until llama-server is ready
3. Start the axum proxy server on the configured host:port
4. Forward signals (SIGINT/SIGTERM) to the child process for clean shutdown

**For `llama run`:**
1. Spawn `llama-cli` with `--conversation` flag
2. Connect stdin/stdout/stderr to the terminal
3. Wait for process exit

Flag construction mirrors the shell script's `build_common_flags()` function.

### 4. API Proxy Layer (`api/`)

An axum HTTP server that sits in front of the llama-server subprocess.

**OpenAI-compatible endpoints** (`/v1/*`):
- `/v1/chat/completions` — proxy to llama-server's same endpoint
- `/v1/models` — return model info
- Non-streaming: forward request, await response, relay back
- Streaming: forward request, relay SSE chunks as-is (`data: {json}\n\n` + `data: [DONE]\n\n`)

**Ollama-compatible endpoints** (`/api/*`):
- `/api/chat` — translate request to OpenAI format, proxy to llama-server, translate response back
- `/api/generate` — same pattern for raw completions
- `/api/tags` — list available models (maps to `llama ls` logic)
- `/api/show` — model info
- Non-streaming: collect full response, reformat as Ollama JSON
- Streaming: translate SSE chunks to NDJSON on the fly

### 5. Streaming Architecture

```
llama-server (SSE)  →  token stream  →  OpenAI SSE passthrough
                                     →  Ollama NDJSON translation
```

The proxy reads llama-server's SSE stream and:
- For `/v1/*`: relays SSE events directly (zero translation)
- For `/api/*`: parses each `chat.completion.chunk`, extracts `delta.content`, wraps in Ollama JSON format, writes as NDJSON line

Implementation:
- `reqwest` with streaming response to consume llama-server's SSE output
- `tokio::sync::mpsc` channel to bridge the reqwest stream to axum's response stream
- Axum `Sse<impl Stream>` for OpenAI endpoints
- Axum `Body::from_stream()` for Ollama NDJSON endpoints

### 6. Download Manager (`download/`)

Downloads GGUF files from HuggingFace with parallel chunk downloads.

Usage: `llama pull mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF Q4_K_M.gguf`

Flow:
1. Resolve download URL: `https://huggingface.co/{org}/{repo}/resolve/main/{filename}`
2. Create local directory: `$LLAMA_MODELS_DIR/{org}/{repo}/`
3. HEAD request to get file size
4. Split into chunks, download in parallel (configurable concurrency)
5. Write to temp file, rename on completion
6. Show progress bar with speed and ETA

### 7. Model Manager (`model/`)

Scans `$LLAMA_MODELS_DIR` for `.gguf` files recursively.

- `llama ls`: Walk the directory tree, collect `ModelInfo` (name, size, path, modified date), sort by name, display as table
- `llama rm <model>`: Resolve the model path, check if a llama-server process is using it (by checking pid files or process list), kill if running, delete the file

## Concurrency Model

- **Main thread**: For `llama run`, owns the terminal (stdin/stdout forwarded to llama-cli)
- **Tokio runtime**: For `llama serve`, runs the axum server and manages async proxy streams
- **Child process management**: llama-server/llama-cli run as separate OS processes, communicating via HTTP (serve) or stdio (run)

## Error Handling

- `anyhow::Result` for CLI commands and top-level flows
- `thiserror` enums for typed errors in library modules (`download::Error`, `process::Error`)
- Errors are logged via `tracing` and surfaced to the user with context
- Child process failures include stderr capture for debugging

## What This Is NOT

- Not a full Ollama replacement — subset of API surface, no model registry
- No multi-model concurrent serving — one model per `llama serve` invocation
- No authentication
- No FFI — purely subprocess-based
- No model conversion/quantization — expects pre-quantized GGUF files
