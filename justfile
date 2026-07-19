# llama.rs — baseline + non-live test gate + live test extras

bins    := "llama"
bin_dir := env_var("HOME") / ".local/bin"
sys_dir := "/usr/local/bin"

# List available recipes
default:
    @just --list

# Build release binaries
build:
    cargo build --release

# Run unit and mock tests (no live server needed)
test:
    cargo test --lib --test cli --test api

# Run live integration tests against llama-server on localhost:8080
test-live:
    cargo test --test live_server

# Run signal handling and cancellation tests (needs llama-server)
test-signal:
    cargo test --test live_signal

# Run live download tests against HuggingFace
test-download:
    cargo test --test live_download

# Run ollama CLI compatibility tests (needs llama-server + ollama binary)
test-ollama:
    cargo test --test live_ollama_cli

# Run ALL tests (needs llama-server on :8080 and ollama binary)
test-all:
    cargo test

# Auto-format the tree
fmt:
    cargo fmt --all

# Check formatting (CI gate)
fmt-check:
    cargo fmt --all -- --check

# Lint — warnings denied (CI gate)
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Full local gate, mirrors CI (fmt + clippy + non-live tests)
check: fmt-check lint test

# Compress every release binary with upx (skips a binary if already packed)
compress: build
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v upx >/dev/null 2>&1; then
        echo "compress: upx not found in PATH" >&2
        exit 1
    fi
    for b in {{bins}}; do
        path="target/release/$b"
        if [ ! -f "$path" ]; then
            echo "compress: missing $path (is bins= correct?)" >&2
            exit 1
        fi
        upx -t "$path" >/dev/null 2>&1 || upx --best --lzma "$path"
        echo "compressed $path"
    done

# Install into ~/.local/bin (default) or /usr/local/bin (--system, via sudo)
install *flags: compress
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{bin_dir}}"
    sudo=""
    for f in {{flags}}; do
        case "$f" in
            --system) dir="{{sys_dir}}"; sudo="sudo" ;;
            *) echo "install: unknown flag '$f' (only --system is supported)" >&2; exit 1 ;;
        esac
    done
    for b in {{bins}}; do
        $sudo install -Dm755 "target/release/$b" "$dir/$b"
        echo "installed $dir/$b"
    done

# Remove installed binaries (pass --system for /usr/local/bin via sudo)
uninstall *flags:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{bin_dir}}"
    sudo=""
    for f in {{flags}}; do
        case "$f" in
            --system) dir="{{sys_dir}}"; sudo="sudo" ;;
            *) echo "uninstall: unknown flag '$f' (only --system is supported)" >&2; exit 1 ;;
        esac
    done
    for b in {{bins}}; do
        $sudo rm -f "$dir/$b"
        echo "removed $dir/$b"
    done

# Remove build artifacts
clean:
    cargo clean
