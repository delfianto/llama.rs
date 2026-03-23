# Step 03: Process Manager & `llama run`

## Objective
Implement the process manager for spawning llama.cpp binaries and wire up `llama run` for interactive REPL mode.

## Instructions

### 1. Process Manager (`process/mod.rs`)

Define shared types:
```rust
pub struct ProcessHandle {
    pub child: tokio::process::Child,
    pub pid: u32,
}
```

### 2. CLI Process (`process/cli.rs`)

`spawn_cli(config: &Config, model_path: &Path) -> Result<ProcessHandle>`:
1. Resolve `llama-cli` binary path via `config.find_binary("llama-cli")`
2. Build common flags from config
3. Add REPL-specific flags: `--conversation`, `-p <system_prompt>`, `--color`
4. Spawn with `std::process::Command` (NOT tokio — we need to inherit stdio)
5. Set `stdin`, `stdout`, `stderr` to `Inherited` so the user interacts directly
6. Return the handle

**Important**: Use `std::process::Command` for `llama run` because we need the child to own the terminal. Tokio's async process doesn't play well with interactive terminal apps.

### 3. Wire Up `llama run` (`cli/run.rs`)

```rust
pub fn exec_run(config: &Config, model: &str) -> Result<()> {
    let model_path = resolve_model_path(&config.models_dir, model)?;

    // Print info (matching shell script's info output)
    info!("Model:        {}", model_path.display());
    info!("Mode:         interactive REPL");
    info!("GPU layers:   {}", config.gpu_layers);
    info!("Context size: {}", config.ctx_size);

    let mut handle = spawn_cli(config, &model_path)?;
    let status = handle.child.wait()?;

    if !status.success() {
        anyhow::bail!("llama-cli exited with status {}", status);
    }
    Ok(())
}
```

### 4. Signal Handling

Register a SIGINT handler that forwards the signal to the child process rather than killing the parent immediately. This allows llama-cli to handle Ctrl+C gracefully (it uses it to cancel generation).

Use `ctrlc` crate or `nix::sys::signal` to set up the handler.

### 5. Update main.rs

Wire `Commands::Run { model }` to call `cli::run::exec_run(&config, &model)`.

## Tests

```rust
#[cfg(test)]
mod tests {
    // Test flag construction produces correct arguments
    fn test_cli_flags_contain_conversation() { ... }
    fn test_cli_flags_contain_system_prompt() { ... }
    fn test_cli_flags_contain_color() { ... }

    // Integration test (only if llama-cli is available)
    // Mark with #[ignore] for CI
    #[test]
    #[ignore]
    fn test_run_with_nonexistent_model_errors() { ... }
}
```

## Acceptance Criteria

- [ ] `llama run model.gguf` spawns llama-cli with correct flags
- [ ] User can type and interact with the REPL
- [ ] Ctrl+C is forwarded to llama-cli (not killed abruptly)
- [ ] Clean exit when llama-cli terminates
- [ ] Helpful error if llama-cli binary not found
- [ ] Helpful error if model file not found
- [ ] Quality gate passes
