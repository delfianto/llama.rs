use thiserror::Error;

#[derive(Error, Debug)]
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

    #[error("Port {port} already in use. Set LLAMA_PORT to use a different port")]
    PortInUse { port: u16 },

    #[error("Process error: {0}")]
    Process(#[from] std::io::Error),
}

/// Colored output helpers matching the shell script's style.
/// Use these for user-facing messages; use `tracing` for debug/trace internals.
pub mod output {
    /// Blue `::` prefix — informational.
    pub fn info(msg: &str) {
        eprintln!("\x1b[34m::\x1b[0m {msg}");
    }

    /// Green `::` prefix — success.
    pub fn success(msg: &str) {
        eprintln!("\x1b[32m::\x1b[0m {msg}");
    }

    /// Yellow warning.
    pub fn warn(msg: &str) {
        eprintln!("\x1b[33mWarning:\x1b[0m {msg}");
    }

    /// Red error.
    pub fn error(msg: &str) {
        eprintln!("\x1b[31mError:\x1b[0m {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages_are_helpful() {
        let e = LlamaError::ModelNotFound {
            path: "foo.gguf".into(),
        };
        assert!(e.to_string().contains("foo.gguf"));

        let e = LlamaError::BinaryNotFound {
            name: "llama-server".into(),
        };
        assert!(e.to_string().contains("LLAMA_BIN_DIR"));
        assert!(e.to_string().contains("PATH"));

        let e = LlamaError::HealthTimeout { seconds: 120 };
        assert!(e.to_string().contains("120"));

        let e = LlamaError::HfAccessDenied;
        assert!(e.to_string().contains("HF_TOKEN"));

        let e = LlamaError::PortInUse { port: 8080 };
        assert!(e.to_string().contains("8080"));
        assert!(e.to_string().contains("LLAMA_PORT"));
    }
}
