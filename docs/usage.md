# Usage

This guide covers installing `llama`, managing GGUF files, running an interactive
session, and serving a model. Configuration values and YAML fields are documented
separately in [configuration.md](configuration.md).

## Installation

Install Rust 1.85 or newer and build `llama-server` plus `llama-cli` from either
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) or
[upstream llama.cpp](https://github.com/ggml-org/llama.cpp).

Install `llama.rs` from the repository:

```bash
cargo install --path .
```

The repository's optimized installation recipe targets the local CPU and runs
UPX before installing to `~/.local/bin`:

```bash
just install
```

That recipe requires `just` and UPX. Use `just install --system` to install into
`/usr/local/bin` through `sudo`.

The wrapper searches `PATH` for the engine binaries. If they live in a build
directory, point the wrapper to it:

```bash
export LLAMA_BIN_DIR=/opt/ik_llama.cpp/build/bin
```

For development, use `cargo run -- <command>` instead of `llama <command>`.

## Command overview

```text
llama pull <org/repo:quant>                  Download a GGUF model
llama ls                                     List downloaded models
llama run [MODEL] [-c FILE] [--device DEV]   Start an interactive session
llama serve [MODEL] [-c FILE] [--device DEV] Start the UI and APIs
llama rm <MODEL>                             Remove a downloaded model
```

Use `llama <command> --help` for the concise built-in reference.

## Download models

`pull` accepts a Hugging Face repository and a quantization selector separated
by a colon:

```bash
llama pull mradermacher/Qwen3-8B-GGUF:Q4_K_M
```

The quantization match is case-insensitive. If several GGUF filenames match, the
shortest matching name is selected. Files are stored as:

```text
$LLAMA_MODELS_DIR/<organization>/<repository>/<file>.gguf
```

Alongside the model, the downloader saves repository metadata such as the README,
tokenizer configuration, chat template, and model configuration when available.
Set `HF_TOKEN` for gated repositories. Existing model files require confirmation
before they are replaced.

Tune download concurrency with `LLAMA_DOWNLOAD_CONNECTIONS`; the default is four.

## Refer to a model

`run`, `serve`, and `rm` accept several model forms:

```bash
# Download-style spec: finds a GGUF containing Q4_K_M
llama run mradermacher/Qwen3-8B-GGUF:Q4_K_M

# Absolute path
llama run /mnt/models/qwen3-8b-q4_k_m.gguf

# Path relative to LLAMA_MODELS_DIR
llama run mradermacher/Qwen3-8B-GGUF/qwen3-8b-q4_k_m.gguf

# Exact filename found recursively
llama run qwen3-8b-q4_k_m.gguf

# Repository directory name; prompts if it contains several models
llama run Qwen3-8B-GGUF
```

Projection files whose names start with `mmproj` are excluded when choosing and
listing text models.

## Interactive mode

Start `llama-cli` in conversation mode:

```bash
llama run mradermacher/Qwen3-8B-GGUF:Q4_K_M
```

The child process owns the terminal, including colored output and input handling.
A single `Ctrl+C` cancels the current generation; a second exits. Prompt, sampling,
context, and compute defaults can come from the environment or a profile.

Examples:

```bash
llama run model.gguf --device cpu
llama run model.gguf --device gpu1
llama run --config profiles/qwen3.yaml
llama run another.gguf --config profiles/qwen3.yaml --device gpu0,gpu1
```

The positional model and `--device` override the corresponding profile values.

## Server mode

Start the compatibility proxy and llama.cpp engine:

```bash
llama serve mradermacher/Qwen3-8B-GGUF:Q4_K_M
```

The wrapper starts `llama-server` on a private random loopback port, waits up to
120 seconds for it to become ready, then binds the public address. Engine stdout
and stderr remain attached to the terminal, so model loading and failures are
visible. If the engine exits during startup, `llama` reports its exit status or
signal immediately.

After `Model loaded!` appears:

- Open `http://127.0.0.1:8080/` for llama.cpp's built-in UI.
- Use the same base URL for OpenAI and Ollama-compatible clients.
- Press `Ctrl+C` for graceful shutdown.

See [api.md](api.md) for endpoints and client examples.

## List and remove models

```bash
llama ls
llama rm mradermacher/Qwen3-8B-GGUF:Q4_K_M
```

`ls` recursively displays model GGUF files under `LLAMA_MODELS_DIR`. Before
removing a file, `rm` looks for a `llama-server` process whose command line uses
that exact path, sends it `SIGTERM`, then removes the model and empty parent
directories.

## Troubleshooting

### Engine binary not found

Verify both binaries are executable and either in `PATH` or in `LLAMA_BIN_DIR`:

```bash
command -v llama-server
command -v llama-cli
```

### Server remains in the loading phase

Loading output comes from the engine itself. Increase its verbosity if needed:

```bash
LLAMA_LOG_VERBOSITY=1 llama serve --config profile.yaml
```

The wrapper's own log filter is controlled by `LLAMA_LOG` or `RUST_LOG`.

### CUDA out of memory

The target model, KV cache, compute buffers, and any speculative draft model all
need memory. Reduce one or more of context size, batch size, or GPU offload; adjust
the tensor split; or remove speculative/MTP arguments from `extra_args`:

```bash
LLAMA_CTX_SIZE=8192 LLAMA_BATCH_SIZE=512 llama serve model.gguf
llama serve model.gguf --device cpu
```

The final engine error is authoritative. A later allocator abort or corruption
message can be a secondary failure during llama.cpp cleanup.

### Address already in use

Change `LLAMA_PORT` or the profile's `server.port`. The internal engine port is
chosen automatically and is separate from the public port.
