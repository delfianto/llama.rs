pub mod resolve;

use std::path::{Path, PathBuf};

use crate::error::LlamaError;

/// Immutable application configuration, loaded from environment variables.
pub struct Config {
    // Paths
    pub bin_dir: Option<PathBuf>,
    pub models_dir: PathBuf,

    // GPU / Hardware
    pub gpu_layers: u32,
    pub tensor_split: Option<String>,
    pub main_gpu: u32,
    pub flash_attn: bool,
    pub mlock: bool,

    // Inference
    pub ctx_size: u32,
    pub batch_size: u32,
    pub threads: u32,

    // Server
    pub host: String,
    pub port: u16,

    // REPL
    pub system_prompt: String,

    // Download
    pub download_connections: u8,
    pub hf_token: Option<String>,

    // Logging
    pub log_level: String,
}

impl Config {
    /// Load configuration from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let default_threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);

        let default_models_dir = dirs::data_dir().map_or_else(
            || {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".local/share/llama/models")
            },
            |d| d.join("llama").join("models"),
        );

        Self {
            bin_dir: env_opt("LLAMA_BIN_DIR").map(PathBuf::from),
            models_dir: PathBuf::from(
                env_or("LLAMA_MODELS_DIR", &default_models_dir.to_string_lossy()).as_ref(),
            ),
            gpu_layers: env_parse("LLAMA_GPU_LAYERS", 999),
            tensor_split: env_opt("LLAMA_TENSOR_SPLIT"),
            main_gpu: env_parse("LLAMA_MAIN_GPU", 0),
            flash_attn: env_bool("LLAMA_FLASH_ATTN", true),
            mlock: env_bool("LLAMA_MLOCK", true),
            ctx_size: env_parse("LLAMA_CTX_SIZE", 32768),
            batch_size: env_parse("LLAMA_BATCH_SIZE", 2048),
            threads: env_parse("LLAMA_THREADS", default_threads),
            host: env_or("LLAMA_HOST", "127.0.0.1").to_string(),
            port: env_parse("LLAMA_PORT", 8080),
            system_prompt: env_or("LLAMA_SYSTEM_PROMPT", "You are a helpful assistant.")
                .to_string(),
            download_connections: env_parse("LLAMA_DOWNLOAD_CONNECTIONS", 4),
            hf_token: env_opt("HF_TOKEN"),
            log_level: env_or("LLAMA_LOG", "info").to_string(),
        }
    }

    /// Resolve a llama.cpp binary by name.
    ///
    /// Checks `bin_dir` first, then falls back to PATH search.
    pub fn find_binary(&self, name: &str) -> Result<PathBuf, LlamaError> {
        if let Some(ref dir) = self.bin_dir {
            let path = dir.join(name);
            if path.is_file() {
                return Ok(path);
            }
            return Err(LlamaError::BinaryNotFound {
                name: format!("{} (not found in {})", name, dir.display()),
            });
        }

        which::which(name).map_err(|_| LlamaError::BinaryNotFound {
            name: name.to_string(),
        })
    }

    /// Build common CLI flags for llama-server / llama-cli, mirroring the
    /// shell script's `build_common_flags()`.
    pub fn build_common_flags(&self, model_path: &Path) -> Vec<String> {
        let mut flags = vec![
            "-m".to_string(),
            model_path.to_string_lossy().to_string(),
            "-ngl".to_string(),
            self.gpu_layers.to_string(),
            "--main-gpu".to_string(),
            self.main_gpu.to_string(),
            "-c".to_string(),
            self.ctx_size.to_string(),
            "-b".to_string(),
            self.batch_size.to_string(),
            "-t".to_string(),
            self.threads.to_string(),
        ];

        if let Some(ref ts) = self.tensor_split {
            flags.push("--tensor-split".to_string());
            flags.push(ts.clone());
        }

        if !self.flash_attn {
            flags.push("-fa".to_string());
            flags.push("off".to_string());
        }

        if self.mlock {
            flags.push("--mlock".to_string());
        }

        flags
    }
}

/// Read an env var, returning `None` if unset or empty.
fn env_opt(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// Read an env var with a default fallback.
fn env_or<'a>(key: &str, default: &'a str) -> std::borrow::Cow<'a, str> {
    match env_opt(key) {
        Some(v) => std::borrow::Cow::Owned(v),
        None => std::borrow::Cow::Borrowed(default),
    }
}

/// Read an env var and parse it, falling back to a default on missing or bad parse.
fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    env_opt(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Read an env var as a boolean (`"1"` / `"true"` = true, `"0"` / `"false"` = false).
fn env_bool(key: &str, default: bool) -> bool {
    match env_opt(key).as_deref() {
        Some("1" | "true" | "yes") => true,
        Some("0" | "false" | "no") => false,
        _ => default,
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_defaults_are_sensible() {
        // Clear any env vars that might interfere
        for key in [
            "LLAMA_BIN_DIR",
            "LLAMA_MODELS_DIR",
            "LLAMA_GPU_LAYERS",
            "LLAMA_TENSOR_SPLIT",
            "LLAMA_MAIN_GPU",
            "LLAMA_FLASH_ATTN",
            "LLAMA_MLOCK",
            "LLAMA_CTX_SIZE",
            "LLAMA_BATCH_SIZE",
            "LLAMA_THREADS",
            "LLAMA_HOST",
            "LLAMA_PORT",
            "LLAMA_SYSTEM_PROMPT",
            "LLAMA_DOWNLOAD_CONNECTIONS",
            "LLAMA_LOG",
        ] {
            unsafe { std::env::remove_var(key) };
        }

        let config = Config::from_env();
        assert_eq!(config.gpu_layers, 999);
        assert_eq!(config.main_gpu, 0);
        assert!(config.flash_attn);
        assert!(config.mlock);
        assert_eq!(config.ctx_size, 32768);
        assert_eq!(config.batch_size, 2048);
        assert!(config.threads > 0);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.system_prompt, "You are a helpful assistant.");
        assert_eq!(config.download_connections, 4);
        assert_eq!(config.log_level, "info");
        assert!(config.bin_dir.is_none());
        assert!(config.tensor_split.is_none());
        assert!(config.hf_token.is_none());
    }

    #[test]
    fn test_env_bool_parsing() {
        assert!(env_bool("_TEST_NONEXISTENT_BOOL", true));
        assert!(!env_bool("_TEST_NONEXISTENT_BOOL", false));

        unsafe { std::env::set_var("_TEST_BOOL", "0") };
        assert!(!env_bool("_TEST_BOOL", true));

        unsafe { std::env::set_var("_TEST_BOOL", "1") };
        assert!(env_bool("_TEST_BOOL", false));

        unsafe { std::env::set_var("_TEST_BOOL", "false") };
        assert!(!env_bool("_TEST_BOOL", true));

        unsafe { std::env::set_var("_TEST_BOOL", "true") };
        assert!(env_bool("_TEST_BOOL", false));

        unsafe { std::env::remove_var("_TEST_BOOL") };
    }

    #[test]
    fn test_env_parse() {
        unsafe { std::env::set_var("_TEST_U32", "42") };
        assert_eq!(env_parse::<u32>("_TEST_U32", 0), 42);

        unsafe { std::env::set_var("_TEST_U32", "not_a_number") };
        assert_eq!(env_parse::<u32>("_TEST_U32", 99), 99);

        unsafe { std::env::remove_var("_TEST_U32") };
        assert_eq!(env_parse::<u32>("_TEST_U32", 99), 99);
    }

    #[test]
    fn test_common_flags_basic() {
        unsafe { std::env::remove_var("LLAMA_TENSOR_SPLIT") };
        let mut config = Config::from_env();
        config.gpu_layers = 999;
        config.main_gpu = 0;
        config.ctx_size = 32768;
        config.batch_size = 2048;
        config.threads = 8;
        config.tensor_split = None;
        config.flash_attn = true;
        config.mlock = true;

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));

        assert!(flags.contains(&"-m".to_string()));
        assert!(flags.contains(&"/models/test.gguf".to_string()));
        assert!(flags.contains(&"-ngl".to_string()));
        assert!(flags.contains(&"999".to_string()));
        assert!(flags.contains(&"--mlock".to_string()));
        // flash_attn is on by default, so no -fa flag
        assert!(!flags.contains(&"-fa".to_string()));
        // no tensor-split when None
        assert!(!flags.contains(&"--tensor-split".to_string()));
    }

    #[test]
    fn test_common_flags_with_tensor_split() {
        let mut config = Config::from_env();
        config.tensor_split = Some("14,12".to_string());

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));
        let ts_idx = flags.iter().position(|f| f == "--tensor-split");
        assert!(ts_idx.is_some());
        assert_eq!(flags[ts_idx.expect("just checked") + 1], "14,12");
    }

    #[test]
    fn test_common_flags_flash_attn_off() {
        let mut config = Config::from_env();
        config.flash_attn = false;

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));
        let fa_idx = flags.iter().position(|f| f == "-fa");
        assert!(fa_idx.is_some());
        assert_eq!(flags[fa_idx.expect("just checked") + 1], "off");
    }

    #[test]
    fn test_common_flags_mlock_off() {
        let mut config = Config::from_env();
        config.mlock = false;

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));
        assert!(!flags.contains(&"--mlock".to_string()));
    }

    #[test]
    fn test_find_binary_with_explicit_dir() {
        let config = Config {
            bin_dir: Some(PathBuf::from("/nonexistent/bin")),
            ..Config::from_env()
        };
        let result = config.find_binary("llama-server");
        assert!(result.is_err());
    }
}
