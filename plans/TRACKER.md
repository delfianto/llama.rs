# Implementation Tracker

## Status Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Complete |
| `[!]` | Blocked |

## Phase 1: Foundation

| Step | Description | Status |
|------|-------------|--------|
| STEP01 | Project scaffolding, Cargo.toml, module stubs, CLI skeleton | `[ ]` |
| STEP02 | Configuration system (env vars, defaults, model path resolution) | `[ ]` |

## Phase 2: Core CLI

| Step | Description | Status |
|------|-------------|--------|
| STEP03 | Process manager & `llama run` (interactive REPL) | `[ ]` |
| STEP04 | `llama serve` — spawn llama-server, health check, lifecycle | `[ ]` |

## Phase 3: API Proxy

| Step | Description | Status |
|------|-------------|--------|
| STEP05 | OpenAI API passthrough (non-streaming) | `[ ]` |
| STEP06 | OpenAI SSE streaming passthrough | `[ ]` |
| STEP07 | Ollama API translation (non-streaming) | `[ ]` |
| STEP08 | Ollama NDJSON streaming | `[ ]` |

## Phase 4: Model Management

| Step | Description | Status |
|------|-------------|--------|
| STEP09 | HuggingFace download manager with parallel downloads | `[ ]` |
| STEP10 | Model list (`llama ls`) and delete (`llama rm`) | `[ ]` |

## Phase 5: Polish

| Step | Description | Status |
|------|-------------|--------|
| STEP11 | Error handling, logging, colored output, graceful shutdown | `[ ]` |
| STEP12 | Integration testing & end-to-end validation | `[ ]` |

## Summary

| Phase | Steps | Complete |
|-------|-------|----------|
| Foundation | 2 | 0 |
| Core CLI | 2 | 0 |
| API Proxy | 4 | 0 |
| Model Management | 2 | 0 |
| Polish | 2 | 0 |
| **Total** | **12** | **0** |

## Quality Gate

Every step must pass before moving to the next:
```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test
```
