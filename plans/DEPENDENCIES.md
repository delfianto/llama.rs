# Dependencies

## Rust Edition & Toolchain

- **Edition**: 2021
- **MSRV**: 1.75.0 (for async trait stabilization)

## Cargo.toml

```toml
[package]
name = "llama-rs"
version = "0.1.0"
edition = "2021"
rust-version = "1.75.0"
description = "Ollama-like CLI wrapper for llama.cpp"
license = "MIT"

[[bin]]
name = "llama"
path = "src/main.rs"

[dependencies]
# CLI
clap = { version = "4", features = ["derive"] }

# Async runtime
tokio = { version = "1", features = ["full"] }
futures = "0.3"
async-stream = "0.3"

# HTTP server (API proxy)
axum = { version = "0.8", features = ["macros"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }

# HTTP client (proxy to llama-server, HF downloads)
reqwest = { version = "0.12", features = ["stream", "json"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Error handling
anyhow = "1"
thiserror = "2"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Progress display
indicatif = "0.17"

# Utilities
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
bytesize = "1"
nix = { version = "0.29", features = ["signal", "process"] }
tokio-util = { version = "0.7", features = ["io"] }

[dev-dependencies]
tempfile = "3"
assert_cmd = "2"
predicates = "3"
tokio-test = "0.4"
wiremock = "0.6"

[profile.release]
opt-level = 3
lto = "thin"
strip = true

[lints.rust]
unsafe_code = "forbid"

[lints.clippy]
all = { level = "deny", priority = -1 }
pedantic = { level = "warn", priority = -1 }
unwrap_used = "deny"
```

## Dependency Rationale

| Crate | Purpose | Why this one |
|-------|---------|-------------|
| `clap` | CLI parsing | Industry standard, derive macros for clean code |
| `tokio` | Async runtime | Required by axum, reqwest; full-featured |
| `axum` | HTTP server | Tokio-native, first-class SSE support via `axum::response::sse`, ergonomic |
| `reqwest` | HTTP client | Streaming response support, used for both proxy and HF downloads |
| `async-stream` | Stream construction | `stream!` macro for ergonomic async stream creation |
| `serde/serde_json` | Serialization | Standard for JSON handling |
| `anyhow/thiserror` | Errors | `anyhow` for app-level, `thiserror` for typed library errors |
| `tracing` | Logging | Structured, async-aware logging |
| `indicatif` | Progress bars | Download progress display |
| `nix` | Signal handling | POSIX signal forwarding to child processes (Unix only) |
| `wiremock` | Test mocking | Mock llama-server HTTP responses in tests |
| `assert_cmd` | CLI testing | Integration tests for the binary |

## Notable Decisions

- **No `skim`/`dialoguer`**: No interactive TUI pickers — keep it simple like the shell script
- **No `clap_complete`**: Shell completions can be added later if wanted
- **`reqwest` over `hyper` directly**: Higher-level API, simpler streaming, sufficient for proxy use case
- **`nix` for signals**: Platform-specific (Unix) but that's fine — llama.cpp is primarily Linux/macOS
- **No `sysinfo`**: Process detection for `llama rm` uses PID files or simple process scanning, not full sysinfo
