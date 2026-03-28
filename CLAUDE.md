# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llama.rs** is a Rust CLI tool that wraps llama.cpp binaries (`llama-server`, `llama-cli`) in an Ollama-like interface. It spawns llama.cpp as child processes and proxies their APIs, adding an Ollama-compatible layer on top. The goal is seamless integration with frontends like OpenWebUI and LibreChat.

This is **not** a full Ollama replacement. Single-model-per-session, no auth, no clustering.

## Build & Run

```bash
cargo build                          # debug build
cargo build --release                # release build
cargo run -- serve model.gguf        # run server
cargo run -- run model.gguf          # interactive REPL
cargo test                           # all tests
cargo test <name>                    # single test
cargo clippy -- -D warnings          # lint (must pass clean)
cargo fmt --check                    # format check
```

**Quality gate** (must pass before every commit):
```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test
```

## Architecture

See `docs/ARCHITECTURE.md` for full details. Key layers:

```
┌─────────────────────────────────────────────┐
│  CLI (clap)                                 │
│  llama run | serve | pull | ls | rm         │
├─────────────────────────────────────────────┤
│  Config (env vars + defaults)               │
├──────────────┬──────────────────────────────┤
│  Process Mgr │  Download Mgr │  Model Mgr  │
│  (spawn      │  (HuggingFace │  (list/rm    │
│   llama-cpp) │   parallel)   │   GGUFs)     │
├──────────────┴──────────────────────────────┤
│  API Proxy Layer (axum)                     │
│  ├─ /v1/*          OpenAI passthrough (SSE) │
│  └─ /api/*         Ollama translation(NDJSON)│
└─────────────────────────────────────────────┘
```

- **Child process model**: llama-server/llama-cli are spawned as subprocesses, not linked via FFI
- **Axum** for the HTTP server with native SSE support
- **Two API surfaces**: OpenAI-compatible (SSE passthrough from llama-server) and Ollama-compatible (NDJSON translation)
- **tokio::sync::mpsc** channels decouple inference subprocess output from HTTP response streams

## Code Conventions

- Rust 2021 edition, MSRV 1.75+
- `anyhow` for application errors, `thiserror` for library-style errors in public APIs
- All async code on tokio runtime
- No `unwrap()` in non-test code — use `?` or `.expect("reason")`
- No `unsafe` — we spawn processes, not FFI
- `tracing` for structured logging (not `println!`)

## Key Design Decisions

- Env-var-driven config mirrors the original shell script — makes Docker deployments trivial
- Model directory uses HuggingFace org/repo structure: `$LLAMA_MODELS_DIR/org/repo/file.gguf`
- OpenAI API is a direct passthrough to llama-server (no translation needed)
- Ollama API requires format translation (SSE ↔ NDJSON, field name mapping)

## Documentation

Detailed documentation lives in `docs/`:
- `ARCHITECTURE.md` — full system design
- `DEPENDENCIES.md` — crate choices and rationale
- `CONFIG_REFERENCE.md` — all env vars and CLI args
