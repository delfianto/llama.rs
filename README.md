# llama.rs

`llama.rs` is a lightweight launcher and compatibility proxy for
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and
[llama.cpp](https://github.com/ggml-org/llama.cpp). It provides an Ollama-like CLI,
an OpenAI-compatible API, a practical subset of the Ollama API, and access to
llama.cpp's built-in web UI.

The wrapper runs `llama-server` and `llama-cli` as child processes. It does not
link llama.cpp, manage several loaded models, or hide the underlying engine's
output.

## Features

- Run local GGUF models interactively or as an HTTP service.
- Keep model, compute, sampling, and llama.cpp flags in reusable YAML profiles.
- Download GGUF files from Hugging Face with parallel connections.
- Use OpenAI clients, Ollama-compatible clients, or the built-in llama.cpp UI.
- See llama.cpp loading progress and errors directly in the terminal.

## Quick start

Requirements:

- Rust 1.85 or newer
- `llama-server` and `llama-cli` from ik_llama.cpp or current upstream llama.cpp

Install the wrapper and ensure the llama.cpp binaries are in `PATH`:

```bash
cargo install --path .

# If the binaries are elsewhere:
export LLAMA_BIN_DIR=/opt/ik_llama.cpp/build/bin
```

Repository contributors can use `just install` for the native optimized build;
that recipe also requires `just` and UPX.

Download and run a model:

```bash
llama pull mradermacher/Qwen3-8B-GGUF:Q4_K_M
llama run mradermacher/Qwen3-8B-GGUF:Q4_K_M
```

Serve the same model:

```bash
llama serve mradermacher/Qwen3-8B-GGUF:Q4_K_M
```

Once loading completes, open <http://127.0.0.1:8080> for llama.cpp's UI or point
an OpenAI/Ollama-compatible client at the same address.

For repeatable experiments, use a YAML profile:

```bash
llama serve --config examples/ik-llama.yaml
llama run --config examples/ik-llama.yaml
```

See the [complete commented profile](examples/ik-llama.yaml) for every supported
field.

## Documentation

- [Usage](docs/usage.md) — commands, model names, downloads, serving, and troubleshooting
- [Configuration](docs/configuration.md) — precedence, environment variables, profiles, and llama.cpp arguments
- [API](docs/api.md) — web UI, OpenAI endpoints, Ollama compatibility, and examples
- [Architecture](docs/architecture.md) — process model, request flows, and source layout

## Scope

This is a single-model local inference wrapper, not a complete Ollama
replacement. It has no authentication, clustering, rate limiting, model
conversion, or quantization. Keep the default loopback bind unless a trusted
reverse proxy supplies the security controls you need.

## License

MIT
