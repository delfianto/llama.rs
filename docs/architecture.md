# Architecture

`llama.rs` is a process wrapper and HTTP compatibility layer. Inference remains
inside an installed ik_llama.cpp or llama.cpp binary; the Rust application does
not use FFI or implement model execution.

## Runtime shape

```text
CLI/configuration
      |
      +-- llama run ----> llama-cli <----> terminal
      |
      +-- llama serve --> llama-server on random loopback port
                              ^
                              |
                         Axum public proxy
                         |       |       |
                      Web UI  OpenAI  Ollama translation
```

There is one model and one child engine per invocation. The separation keeps the
wrapper independent of llama.cpp's ABI and allows users to replace the binaries
without rebuilding Rust code.

## Source layout

```text
src/
├── main.rs              clap definitions, configuration preparation, dispatch
├── lib.rs               library module exports
├── cli/
│   ├── run.rs           interactive llama-cli lifecycle
│   ├── serve.rs         engine startup and public HTTP lifecycle
│   ├── pull.rs          Hugging Face model selection/download command
│   ├── ls.rs            local model listing
│   └── rm.rs            process detection and model removal
├── config/
│   ├── mod.rs           environment defaults and llama.cpp flag construction
│   ├── yaml.rs          typed YAML profile overlay
│   └── resolve.rs       model-name and path resolution
├── process/
│   ├── cli.rs           llama-cli spawn and signal behavior
│   ├── server.rs        llama-server spawn, shutdown, and child status
│   └── health.rs        upstream readiness requests
├── api/
│   ├── mod.rs           Axum routes and shared state
│   ├── openai.rs        OpenAI passthrough and model response
│   ├── ollama.rs        Ollama request/response translation
│   ├── upstream.rs      generic UI/native-route passthrough
│   ├── types.rs         wire-format types
│   └── stream/          SSE parsing and NDJSON conversion
├── download/            Hugging Face discovery and parallel range downloads
├── model/               model scanning and display metadata
└── error.rs             application errors and terminal output helpers
```

## Configuration flow

Startup creates a `Config` from defaults and environment variables. For `run` and
`serve`, a typed YAML profile may overlay it. Finally, explicit CLI values replace
the model and device. Unknown profile fields are rejected by Serde.

The resolved configuration builds an argument vector rather than a shell command.
This avoids shell interpolation and preserves exact `extra_args` boundaries.

See [configuration.md](configuration.md) for the public contract.

## Interactive process lifecycle

`llama run` resolves the GGUF and starts `llama-cli` with inherited stdin, stdout,
and stderr. The wrapper adds conversation mode, system prompt, colors, optional
reverse prompts, and the common compute/sampling flags.

The child owns normal terminal interaction. Signal handling allows one `Ctrl+C`
to interrupt generation while a second exits the session.

## Server process lifecycle

`llama serve` performs these steps:

1. Resolve the selected model and `llama-server` binary.
2. Reserve a random loopback port for the private engine.
3. Spawn `llama-server` with inherited stdout and stderr.
4. Race health polling against child-process exit for up to 120 seconds.
5. Build shared API state and bind the configured public address.
6. On `Ctrl+C`, stop Axum and terminate the child gracefully, escalating if needed.

The public listener is deliberately created after readiness. Clients therefore
cannot reach a half-loaded proxy, and immediate engine failures include the actual
exit status or signal.

## HTTP request flows

### Native UI and engine routes

`GET /` and unmatched paths are generic reverse-proxy requests to the private
llama-server. Status, headers, and streaming bodies are relayed, which exposes the
built-in UI and avoids maintaining a duplicate list of native endpoints.

`HEAD /` is handled locally because Ollama clients use it as a connectivity check.

### OpenAI compatibility

`POST /v1/chat/completions` forwards the raw request body to the same upstream
route and streams the response body back. The wrapper does not deserialize this
path, so extra fields supported by a particular llama.cpp build survive intact.

`GET /v1/models` is synthesized locally because the wrapper knows the single
loaded model.

### Ollama compatibility

`/api/chat` and `/api/generate` deserialize the supported Ollama request subset,
map it to an OpenAI chat request, then translate the response. Non-streaming calls
produce one JSON object. Streaming calls parse upstream SSE events and emit Ollama
NDJSON incrementally.

Metadata routes such as `/api/tags`, `/api/show`, and `/api/version` are generated
locally from the loaded model and wrapper information.

See [api.md](api.md) for the externally supported routes.

## Model and download management

Models use the Hugging Face organization/repository directory convention. The
resolver supports absolute paths, repository-relative paths, recursive filename
search, repository directory selection, and `org/repo:quant` specs. Auxiliary
`mmproj` GGUF files are excluded from normal model selection.

The downloader queries the Hugging Face API, selects a quantization-matching GGUF,
and uses parallel HTTP range requests when supported. A temporary download is
renamed only after completion. Repository metadata is fetched separately on a
best-effort basis.

Removal resolves the exact model path, detects matching `llama-server` command
lines, terminates them, deletes the file, and cleans empty repository directories.

## Concurrency and errors

The Tokio runtime owns HTTP serving, upstream requests, downloads, health polling,
and child-process control. Response bodies are streamed to avoid buffering model
output. The interactive child instead uses inherited blocking terminal handles.

Top-level commands return `anyhow::Result` with contextual errors. Reusable error
conditions use typed `thiserror` variants. Structured wrapper diagnostics use
`tracing`; user-facing status uses the terminal output helpers; child engine logs
are inherited verbatim.

## Deliberate limits

- No FFI or embedded inference engine
- No authentication, authorization, or rate limiting
- No multi-model scheduler or concurrent model registry
- No model conversion, quantization, or Modelfile implementation
- Ollama compatibility is a targeted subset, not protocol parity
