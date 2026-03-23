# Step 04: `llama serve` — Spawn llama-server

## Objective
Implement spawning llama-server as a child process, health-check polling, and basic lifecycle management. No proxy yet — just get the subprocess running and healthy.

## Instructions

### 1. Server Process (`process/server.rs`)

`spawn_server(config: &Config, model_path: &Path) -> Result<ProcessHandle>`:
1. Resolve `llama-server` binary path
2. Build common flags
3. Add server-specific flags: `--host`, `--port`, `--metrics`
4. **Important**: llama-server binds to the port itself. Our axum proxy will use a DIFFERENT port. So we need an internal port for llama-server and the user-facing port for our proxy.
   - llama-server listens on `127.0.0.1:{internal_port}` (e.g., configured port + 1, or a random free port)
   - Our axum proxy listens on `{LLAMA_HOST}:{LLAMA_PORT}`
5. Spawn with `tokio::process::Command` (async — we don't need terminal inheritance)
6. Capture stdout/stderr for logging (pipe to tracing)

### 2. Internal Port Selection

`fn find_free_port() -> Result<u16>`:
- Bind a `TcpListener` to `:0`, get the assigned port, drop the listener
- This gives us a guaranteed-free port for llama-server's internal API

Store the internal port in a `ServerState` struct alongside the `ProcessHandle`.

### 3. Health Check (`process/health.rs`)

`async fn wait_for_ready(url: &str, timeout: Duration) -> Result<()>`:
1. Poll `GET {url}/health` every 500ms
2. llama-server returns 200 when ready
3. Timeout after configurable duration (default 120s — large models take time to load)
4. Log progress: "Waiting for llama-server to load model..."
5. If the child process exits during polling, detect and report the error immediately

### 4. Wire Up `llama serve` (`cli/serve.rs`) — Phase 1

```rust
pub async fn exec_serve(config: Arc<Config>, model: &str) -> Result<()> {
    let model_path = resolve_model_path(&config.models_dir, model)?;

    // Print info
    info!("Model:        {}", model_path.display());
    info!("Endpoint:     http://{}:{}", config.host, config.port);
    // ... more info lines matching shell script

    let server_state = spawn_server(&config, &model_path).await?;
    info!("Waiting for model to load...");
    wait_for_ready(&server_state.internal_url, Duration::from_secs(120)).await?;
    info!("Model loaded, server ready!");

    // TODO (Step 05): Start axum proxy here
    // For now, just wait for the child process
    server_state.handle.child.wait().await?;
    Ok(())
}
```

### 5. Graceful Shutdown

On SIGINT/SIGTERM:
1. Send SIGTERM to the llama-server child process
2. Wait up to 5s for clean exit
3. SIGKILL if still running
4. Exit

Use `tokio::signal::ctrl_c()` for the trigger.

### 6. Stderr Logging

Spawn a background task that reads llama-server's stderr line by line and logs via `tracing::debug!`. This ensures model loading progress and any errors are visible.

## Tests

```rust
#[cfg(test)]
mod tests {
    // Test internal port selection
    fn test_find_free_port_returns_valid_port() { ... }
    fn test_find_free_port_returns_different_ports() { ... }

    // Test health check with mock server (wiremock)
    async fn test_health_check_succeeds_on_200() { ... }
    async fn test_health_check_retries_on_failure() { ... }
    async fn test_health_check_times_out() { ... }

    // Test server flag construction
    fn test_server_flags_contain_host_port() { ... }
    fn test_server_flags_contain_metrics() { ... }
}
```

## Acceptance Criteria

- [ ] `llama serve model.gguf` spawns llama-server on an internal port
- [ ] Health check waits for model loading
- [ ] llama-server stderr is captured and logged
- [ ] Ctrl+C cleanly shuts down llama-server
- [ ] Error if llama-server binary not found
- [ ] Error if model not found
- [ ] Quality gate passes
