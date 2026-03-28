use std::path::Path;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, warn};

use super::AsyncProcessHandle;
use crate::config::Config;

/// State for a running llama-server instance.
pub struct ServerState {
    pub handle: AsyncProcessHandle,
    /// Base URL for the internal llama-server (e.g., `http://127.0.0.1:54321`).
    pub internal_url: String,
    /// The internal port llama-server is listening on.
    pub internal_port: u16,
}

/// Find a free TCP port by binding to `:0` and reading the assigned port.
pub fn find_free_port() -> anyhow::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Spawn llama-server as an async child process on a random free port.
///
/// Captures stderr and forwards it to `tracing::debug!`.
pub async fn spawn_server(config: &Config, model_path: &Path) -> anyhow::Result<ServerState> {
    let binary = config.find_binary("llama-server")?;
    let internal_port = find_free_port()?;
    let internal_url = format!("http://127.0.0.1:{internal_port}");

    let mut flags = config.build_common_flags(model_path);
    flags.extend([
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        internal_port.to_string(),
        "--metrics".to_string(),
    ]);

    debug!("Spawning: {} {}", binary.display(), flags.join(" "));

    let mut cmd = Command::new(&binary);
    cmd.args(&flags)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    // Suppress verbose llama.cpp logging unless user explicitly set it
    if std::env::var("LLAMA_LOG_VERBOSITY").is_err() {
        cmd.env("LLAMA_LOG_VERBOSITY", "0");
    }

    let mut child = cmd.spawn()?;

    let pid = child.id().unwrap_or(0);

    // Spawn background task to log stderr
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                debug!(target: "llama_server", "{line}");
            }
        });
    }

    // Spawn background task to log stdout
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                debug!(target: "llama_server", "{line}");
            }
        });
    }

    Ok(ServerState {
        handle: AsyncProcessHandle { child, pid },
        internal_url,
        internal_port,
    })
}

/// Build the full argument list that `spawn_server` would use (for testing).
pub fn build_server_args(config: &Config, model_path: &Path, port: u16) -> Vec<String> {
    let mut flags = config.build_common_flags(model_path);
    flags.extend([
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--metrics".to_string(),
    ]);
    flags
}

/// Send SIGTERM to the server, wait up to `timeout` for exit, then SIGKILL.
pub async fn shutdown_server(state: &mut ServerState, timeout: std::time::Duration) {
    let pid = state.handle.pid;
    if pid == 0 {
        return;
    }

    // Send SIGTERM
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        #[allow(clippy::cast_possible_wrap)]
        let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
    }

    // Wait for clean exit with timeout
    let wait_result = tokio::time::timeout(timeout, state.handle.child.wait()).await;

    if let Ok(Ok(_)) = wait_result {
        debug!("llama-server exited cleanly");
    } else {
        warn!("llama-server did not exit in time, sending SIGKILL");
        let _ = state.handle.child.kill().await;
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use std::path::Path;

    fn test_config() -> Config {
        for key in [
            "LLAMA_BIN_DIR",
            "LLAMA_TENSOR_SPLIT",
            "LLAMA_SYSTEM_PROMPT_FILE",
            "LLAMA_PROMPT_TEMPLATE_FILE",
            "LLAMA_PROMPT_TEMPLATE",
            "LLAMA_TEMPERATURE",
            "LLAMA_MAX_TOKENS",
            "LLAMA_CTX_OVERFLOW",
            "LLAMA_STOP",
            "LLAMA_TOP_K",
            "LLAMA_REPEAT_PENALTY",
            "LLAMA_PRESENCE_PENALTY",
            "LLAMA_TOP_P",
            "LLAMA_MIN_P",
        ] {
            unsafe { std::env::remove_var(key) };
        }
        let mut config = Config::from_env();
        config.gpu_layers = 999;
        config.ctx_size = 32768;
        config.batch_size = 2048;
        config.threads = 8;
        config.tensor_split = None;
        config.flash_attn = true;
        config.mlock = true;
        config.chat_template = None;
        config.temperature = None;
        config.max_tokens = None;
        config.ctx_overflow = "shift".to_string();
        config.stop = vec![];
        config.top_k = None;
        config.repeat_penalty = None;
        config.presence_penalty = None;
        config.top_p = None;
        config.min_p = None;
        config
    }

    #[test]
    fn test_find_free_port_returns_valid_port() {
        let port = find_free_port().expect("should find free port");
        assert!(port > 0);
    }

    #[test]
    fn test_find_free_port_returns_different_ports() {
        let p1 = find_free_port().expect("port 1");
        let p2 = find_free_port().expect("port 2");
        // Not guaranteed but extremely likely with random OS assignment
        // Just check both are valid
        assert!(p1 > 0);
        assert!(p2 > 0);
    }

    #[test]
    fn test_server_flags_contain_host_port() {
        let config = test_config();
        let flags = build_server_args(&config, Path::new("/models/test.gguf"), 12345);

        let host_idx = flags
            .iter()
            .position(|f| f == "--host")
            .expect("has --host");
        assert_eq!(flags[host_idx + 1], "127.0.0.1");

        let port_idx = flags
            .iter()
            .position(|f| f == "--port")
            .expect("has --port");
        assert_eq!(flags[port_idx + 1], "12345");
    }

    #[test]
    fn test_server_flags_contain_metrics() {
        let config = test_config();
        let flags = build_server_args(&config, Path::new("/models/test.gguf"), 8080);
        assert!(flags.contains(&"--metrics".to_string()));
    }

    #[test]
    fn test_server_flags_contain_model() {
        let config = test_config();
        let flags = build_server_args(&config, Path::new("/models/test.gguf"), 8080);
        let m_idx = flags.iter().position(|f| f == "-m").expect("has -m");
        assert_eq!(flags[m_idx + 1], "/models/test.gguf");
    }
}
