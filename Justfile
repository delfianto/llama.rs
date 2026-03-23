# llama.rs — because ollama kept hurting our feelings

default:
    @just --list

# Build debug binary
build:
    cargo build

# Build release binary
release:
    cargo build --release

# Run quality gate (format check + clippy + tests)
check:
    cargo fmt --check
    cargo clippy -- -D warnings
    cargo test --lib --test cli --test api

# Format code
fmt:
    cargo fmt

# Lint
lint:
    cargo clippy -- -D warnings

# Run unit and mock tests (no live server needed)
test:
    cargo test --lib --test cli --test api

# Run live integration tests against llama-server on localhost:8080
test-live:
    cargo test --test live_server

# Run live download tests against HuggingFace
test-download:
    cargo test --test live_download

# Run ollama CLI compatibility tests (needs llama-server + ollama binary)
test-ollama:
    cargo test --test live_ollama_cli

# Run ALL tests (needs llama-server on :8080 and ollama binary)
test-all:
    cargo test

# Install release binary to ~/.local/bin
install: release
    mkdir -p ~/.local/bin
    cp target/release/llama ~/.local/bin/llama
    @echo "Installed to ~/.local/bin/llama"
    @echo "Make sure ~/.local/bin is in your PATH"

# Uninstall
uninstall:
    rm -f ~/.local/bin/llama
    @echo "Removed ~/.local/bin/llama"

# Clean build artifacts
clean:
    cargo clean
