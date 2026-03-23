# Step 01: Project Scaffolding

## Objective
Set up the complete Rust project skeleton with all modules stubbed, Cargo.toml configured, and `llama --help` working. Zero warnings, zero errors.

## Instructions

### 1. Initialize Project
```bash
cargo init --name llama-rs
```

### 2. Write Cargo.toml
Use the exact dependency list from `DEPENDENCIES.md`. Binary name is `llama`.

### 3. Create Module Structure

Create all source files with minimal stubs (empty `pub mod` declarations or placeholder structs):

```
src/
├── main.rs
├── cli/
│   ├── mod.rs
│   ├── run.rs
│   ├── serve.rs
│   ├── pull.rs
│   ├── ls.rs
│   └── rm.rs
├── config/
│   ├── mod.rs
│   └── resolve.rs
├── process/
│   ├── mod.rs
│   ├── server.rs
│   ├── cli.rs
│   └── health.rs
├── api/
│   ├── mod.rs
│   ├── openai.rs
│   ├── ollama.rs
│   ├── stream/
│   │   ├── mod.rs
│   │   ├── sse.rs
│   │   └── ndjson.rs
│   └── types.rs
├── download/
│   ├── mod.rs
│   ├── hf.rs
│   └── progress.rs
├── model/
│   ├── mod.rs
│   └── types.rs
└── error.rs
```

### 4. Implement main.rs

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llama", version, about = "Ollama-like CLI wrapper for llama.cpp")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive REPL with a model
    Run {
        /// Model file (filename, relative path, or absolute path)
        model: String,
    },
    /// Start API server with a model
    Serve {
        /// Model file (filename, relative path, or absolute path)
        model: String,
    },
    /// Download a GGUF model from HuggingFace
    Pull {
        /// HuggingFace repo (e.g., mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF)
        repo: String,
        /// GGUF filename to download
        filename: String,
    },
    /// List downloaded models
    Ls,
    /// Delete a downloaded model
    Rm {
        /// Model to delete
        model: String,
    },
}
```

Set up tokio runtime, tracing subscriber, load config, dispatch to command handlers (which are all `todo!()` stubs at this point).

### 5. Stub All Modules

Each module file should have at minimum:
- `mod.rs` files declare sub-modules
- Leaf files have a placeholder public function or struct
- Everything compiles and passes `cargo clippy`

## Tests

```rust
#[cfg(test)]
mod tests {
    use assert_cmd::Command;

    #[test]
    fn test_help_flag() {
        Command::cargo_bin("llama").unwrap()
            .arg("--help")
            .assert()
            .success()
            .stdout(predicates::str::contains("Ollama-like CLI wrapper"));
    }

    #[test]
    fn test_version_flag() {
        Command::cargo_bin("llama").unwrap()
            .arg("--version")
            .assert()
            .success();
    }

    #[test]
    fn test_no_args_shows_help() {
        // clap should show help/error when no subcommand given
        Command::cargo_bin("llama").unwrap()
            .assert()
            .failure();
    }
}
```

## Acceptance Criteria

- [ ] `cargo build` succeeds with zero warnings
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo fmt --check` passes
- [ ] `llama --help` shows usage with all 5 subcommands
- [ ] `llama --version` shows version
- [ ] All module files exist and are reachable from `main.rs`
- [ ] Integration tests pass
