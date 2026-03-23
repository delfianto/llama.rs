# Step 11: Error Handling, Logging & Polish

## Objective
Harden error handling across all modules, set up structured logging, add colored CLI output, and overall polish for a good user experience.

## Instructions

### 1. Error Types (`error.rs`)

Define typed errors for each domain:

```rust
#[derive(thiserror::Error, Debug)]
pub enum LlamaError {
    #[error("Model not found: {path}")]
    ModelNotFound { path: String },

    #[error("Binary not found: {name} (set LLAMA_BIN_DIR or add to PATH)")]
    BinaryNotFound { name: String },

    #[error("llama-server failed to start: {reason}")]
    ServerStartFailed { reason: String },

    #[error("llama-server health check timed out after {seconds}s")]
    HealthTimeout { seconds: u64 },

    #[error("Download failed: {reason}")]
    DownloadFailed { reason: String },

    #[error("HuggingFace access denied — set HF_TOKEN for gated models")]
    HfAccessDenied,

    #[error("Process error: {0}")]
    Process(#[from] std::io::Error),
}
```

Replace generic `anyhow::bail!` calls in critical paths with these typed errors for better user messages.

### 2. Logging Setup

In `main.rs`, configure `tracing_subscriber`:

```rust
fn init_logging(level: &str) {
    let filter = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| level.to_string());

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .init();
}
```

- Default to `LLAMA_LOG` env var level
- `RUST_LOG` overrides for fine-grained control
- Keep output clean for users (no thread IDs, no targets at info level)

### 3. Colored Output

Match the shell script's colored output style:
```rust
use std::io::Write;

pub fn info(msg: &str) {
    eprintln!("\x1b[34m::\x1b[0m {}", msg);
}

pub fn success(msg: &str) {
    eprintln!("\x1b[32m::\x1b[0m {}", msg);
}

pub fn warn(msg: &str) {
    eprintln!("\x1b[33mWarning:\x1b[0m {}", msg);
}

pub fn error(msg: &str) {
    eprintln!("\x1b[31mError:\x1b[0m {}", msg);
}
```

Use these for user-facing output. Use `tracing` for debug/trace level internals.

### 4. Startup Banner

When `llama serve` starts, print a config summary matching the shell script:
```
:: Model:         /path/to/model.gguf
:: Endpoint:      http://127.0.0.1:8080
:: GPU layers:    999
:: Tensor split:  14,12
:: Context size:  32768
:: Flash attn:    on
:: Mlock:         on
```

### 5. Graceful Error Messages

Ensure every user-facing error includes actionable guidance:
- "Model not found: X. Run 'llama ls' to see available models or 'llama pull' to download."
- "llama-server not found. Set LLAMA_BIN_DIR or install llama.cpp."
- "Port 8080 already in use. Set LLAMA_PORT to use a different port."

### 6. Ctrl+C Handling

Ensure clean shutdown in all modes:
- `llama serve`: Stop axum, then stop llama-server child process
- `llama run`: Forward signal to llama-cli
- `llama pull`: Clean up partial downloads

## Tests

```rust
#[cfg(test)]
mod tests {
    fn test_error_messages_are_helpful() { ... }
    fn test_config_display_format() { ... }

    // Integration
    #[test]
    fn test_missing_binary_shows_helpful_error() {
        Command::cargo_bin("llama").unwrap()
            .env("LLAMA_BIN_DIR", "/nonexistent")
            .args(["run", "model.gguf"])
            .assert()
            .failure()
            .stderr(predicates::str::contains("not found"));
    }
}
```

## Acceptance Criteria

- [ ] All error paths produce helpful, actionable messages
- [ ] Colored output matches the shell script's style
- [ ] `LLAMA_LOG` and `RUST_LOG` control log verbosity
- [ ] Startup banner shows full config summary
- [ ] Clean Ctrl+C handling in all modes
- [ ] No panics in any error path (all `unwrap` replaced)
- [ ] Quality gate passes
