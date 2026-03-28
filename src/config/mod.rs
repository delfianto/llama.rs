pub mod resolve;

use std::path::{Path, PathBuf};

use crate::error::LlamaError;

/// Resolved chat template source.
#[derive(Debug, Clone)]
pub enum ChatTemplate {
    /// Pass as `--chat-template-file <path>` (file verified to exist).
    File(PathBuf),
    /// Pass as `--chat-template <value>`.
    Value(String),
}

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

    // REPL / Prompt
    pub system_prompt: String,
    pub chat_template: Option<ChatTemplate>,

    // Sampling defaults (None = use llama.cpp default)
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub ctx_overflow: String,
    pub stop: Vec<String>,
    pub top_k: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,

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
            system_prompt: {
                let from_file =
                    env_opt("LLAMA_SYSTEM_PROMPT_FILE").and_then(|path| read_file_or_warn(&path));
                from_file.unwrap_or_else(|| {
                    env_or("LLAMA_SYSTEM_PROMPT", "You are a helpful assistant.").to_string()
                })
            },
            chat_template: {
                let file_path = env_opt("LLAMA_PROMPT_TEMPLATE_FILE");
                let string_val = env_opt("LLAMA_PROMPT_TEMPLATE");
                if let Some(path) = file_path {
                    let p = PathBuf::from(&path);
                    if p.is_file() {
                        Some(ChatTemplate::File(p))
                    } else {
                        tracing::warn!("Chat template file not found: {path}");
                        string_val.map(ChatTemplate::Value)
                    }
                } else {
                    string_val.map(ChatTemplate::Value)
                }
            },
            temperature: env_opt("LLAMA_TEMPERATURE").and_then(|v| v.parse().ok()),
            max_tokens: env_opt("LLAMA_MAX_TOKENS").and_then(|v| v.parse().ok()),
            ctx_overflow: env_or("LLAMA_CTX_OVERFLOW", "shift").to_string(),
            stop: env_opt("LLAMA_STOP")
                .map(|s| {
                    s.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default(),
            top_k: env_opt("LLAMA_TOP_K").and_then(|v| v.parse().ok()),
            repeat_penalty: env_opt("LLAMA_REPEAT_PENALTY").and_then(|v| v.parse().ok()),
            presence_penalty: env_opt("LLAMA_PRESENCE_PENALTY").and_then(|v| v.parse().ok()),
            top_p: env_opt("LLAMA_TOP_P").and_then(|v| v.parse().ok()),
            min_p: env_opt("LLAMA_MIN_P").and_then(|v| v.parse().ok()),
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

        // Sampling defaults
        if let Some(temp) = self.temperature {
            flags.push("--temp".to_string());
            flags.push(temp.to_string());
        }
        if let Some(max) = self.max_tokens {
            flags.push("-n".to_string());
            flags.push(max.to_string());
        }
        if self.ctx_overflow == "stop" {
            flags.push("--no-context-shift".to_string());
        }
        if let Some(k) = self.top_k {
            flags.push("--top-k".to_string());
            flags.push(k.to_string());
        }
        if let Some(rp) = self.repeat_penalty {
            flags.push("--repeat-penalty".to_string());
            flags.push(rp.to_string());
        }
        if let Some(pp) = self.presence_penalty {
            flags.push("--presence-penalty".to_string());
            flags.push(pp.to_string());
        }
        if let Some(p) = self.top_p {
            flags.push("--top-p".to_string());
            flags.push(p.to_string());
        }
        if let Some(mp) = self.min_p {
            flags.push("--min-p".to_string());
            flags.push(mp.to_string());
        }

        // Chat template
        match &self.chat_template {
            Some(ChatTemplate::File(path)) => {
                flags.push("--chat-template-file".to_string());
                flags.push(path.to_string_lossy().to_string());
            }
            Some(ChatTemplate::Value(val)) => {
                flags.push("--chat-template".to_string());
                flags.push(val.clone());
            }
            None => {}
        }

        flags
    }
}

/// Attempt to read a file at `path`. Returns trimmed contents on success,
/// logs a warning and returns `None` on failure.
fn read_file_or_warn(path: &str) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(contents) => Some(contents.trim().to_string()),
        Err(e) => {
            tracing::warn!("Could not read file {path}: {e}");
            None
        }
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

    /// All env var keys that Config reads from.
    const ALL_CONFIG_KEYS: &[&str] = &[
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
        "LLAMA_DOWNLOAD_CONNECTIONS",
        "LLAMA_LOG",
    ];

    fn clear_env() {
        for key in ALL_CONFIG_KEYS {
            unsafe { std::env::remove_var(key) };
        }
    }

    #[test]
    fn test_defaults_are_sensible() {
        clear_env();

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

        // New defaults
        assert!(config.chat_template.is_none());
        assert!(config.temperature.is_none());
        assert!(config.max_tokens.is_none());
        assert_eq!(config.ctx_overflow, "shift");
        assert!(config.stop.is_empty());
        assert!(config.top_k.is_none());
        assert!(config.repeat_penalty.is_none());
        assert!(config.presence_penalty.is_none());
        assert!(config.top_p.is_none());
        assert!(config.min_p.is_none());
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

    // ─── System prompt from file tests ──────────────────────────────────────
    // Combined into one test to avoid env var race conditions across parallel tests.

    #[test]
    fn test_system_prompt_file_loading() {
        clear_env();

        // 1. File exists → use file content (trimmed)
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        std::fs::write(tmp.path(), "You are a pirate.\n").expect("write");
        unsafe { std::env::set_var("LLAMA_SYSTEM_PROMPT_FILE", tmp.path().to_str().unwrap()) };
        let config = Config::from_env();
        assert_eq!(config.system_prompt, "You are a pirate.");

        // 2. File exists AND LLAMA_SYSTEM_PROMPT set → file wins
        unsafe { std::env::set_var("LLAMA_SYSTEM_PROMPT", "From env") };
        let config = Config::from_env();
        assert_eq!(config.system_prompt, "You are a pirate.");

        // 3. File missing → falls back to LLAMA_SYSTEM_PROMPT
        unsafe { std::env::set_var("LLAMA_SYSTEM_PROMPT_FILE", "/nonexistent/prompt.txt") };
        unsafe { std::env::set_var("LLAMA_SYSTEM_PROMPT", "Custom prompt") };
        let config = Config::from_env();
        assert_eq!(config.system_prompt, "Custom prompt");

        // 4. File missing, no LLAMA_SYSTEM_PROMPT → default
        unsafe { std::env::remove_var("LLAMA_SYSTEM_PROMPT") };
        let config = Config::from_env();
        assert_eq!(config.system_prompt, "You are a helpful assistant.");

        unsafe { std::env::remove_var("LLAMA_SYSTEM_PROMPT_FILE") };
    }

    // ─── Prompt template tests ──────────────────────────────────────────────
    // Combined into one test to avoid env var race conditions across parallel tests.

    #[test]
    fn test_prompt_template_loading() {
        clear_env();

        // 1. Neither set → None
        let config = Config::from_env();
        assert!(config.chat_template.is_none());

        // 2. String only → ChatTemplate::Value
        unsafe { std::env::set_var("LLAMA_PROMPT_TEMPLATE", "llama3") };
        let config = Config::from_env();
        assert!(matches!(config.chat_template, Some(ChatTemplate::Value(ref v)) if v == "llama3"));

        // 3. File exists → ChatTemplate::File
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        std::fs::write(tmp.path(), "{% for m in messages %}...{% endfor %}").expect("write");
        unsafe { std::env::set_var("LLAMA_PROMPT_TEMPLATE_FILE", tmp.path().to_str().unwrap()) };
        let config = Config::from_env();
        assert!(matches!(config.chat_template, Some(ChatTemplate::File(_))));

        // 4. File missing → falls back to string value
        unsafe { std::env::set_var("LLAMA_PROMPT_TEMPLATE_FILE", "/nonexistent/template.jinja") };
        unsafe { std::env::set_var("LLAMA_PROMPT_TEMPLATE", "chatml") };
        let config = Config::from_env();
        assert!(matches!(config.chat_template, Some(ChatTemplate::Value(ref v)) if v == "chatml"));

        unsafe { std::env::remove_var("LLAMA_PROMPT_TEMPLATE_FILE") };
        unsafe { std::env::remove_var("LLAMA_PROMPT_TEMPLATE") };
    }

    // ─── Sampling flags tests ───────────────────────────────────────────────

    #[test]
    fn test_common_flags_with_sampling_params() {
        clear_env();
        let mut config = Config::from_env();
        config.temperature = Some(0.7);
        config.max_tokens = Some(2048);
        config.top_k = Some(40);
        config.repeat_penalty = Some(1.1);
        config.presence_penalty = Some(0.5);
        config.top_p = Some(0.9);
        config.min_p = Some(0.05);

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));

        let check_flag = |name: &str, expected: &str| {
            let idx = flags.iter().position(|f| f == name).unwrap_or_else(|| {
                panic!("expected flag {name} in flags: {flags:?}");
            });
            assert_eq!(flags[idx + 1], expected, "flag {name}");
        };

        check_flag("--temp", "0.7");
        check_flag("-n", "2048");
        check_flag("--top-k", "40");
        check_flag("--repeat-penalty", "1.1");
        check_flag("--presence-penalty", "0.5");
        check_flag("--top-p", "0.9");
        check_flag("--min-p", "0.05");
    }

    #[test]
    fn test_common_flags_no_sampling_when_none() {
        clear_env();
        let config = Config::from_env();
        let flags = config.build_common_flags(Path::new("/models/test.gguf"));

        assert!(!flags.contains(&"--temp".to_string()));
        assert!(!flags.contains(&"-n".to_string()));
        assert!(!flags.contains(&"--top-k".to_string()));
        assert!(!flags.contains(&"--repeat-penalty".to_string()));
        assert!(!flags.contains(&"--presence-penalty".to_string()));
        assert!(!flags.contains(&"--top-p".to_string()));
        assert!(!flags.contains(&"--min-p".to_string()));
        assert!(!flags.contains(&"--no-context-shift".to_string()));
    }

    #[test]
    fn test_common_flags_no_context_shift() {
        clear_env();
        let mut config = Config::from_env();
        config.ctx_overflow = "stop".to_string();

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));
        assert!(flags.contains(&"--no-context-shift".to_string()));
    }

    #[test]
    fn test_common_flags_chat_template_file() {
        clear_env();
        let mut config = Config::from_env();
        config.chat_template = Some(ChatTemplate::File(PathBuf::from("/tmp/template.jinja")));

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));
        let idx = flags
            .iter()
            .position(|f| f == "--chat-template-file")
            .expect("has --chat-template-file");
        assert_eq!(flags[idx + 1], "/tmp/template.jinja");
    }

    #[test]
    fn test_common_flags_chat_template_string() {
        clear_env();
        let mut config = Config::from_env();
        config.chat_template = Some(ChatTemplate::Value("chatml".to_string()));

        let flags = config.build_common_flags(Path::new("/models/test.gguf"));
        let idx = flags
            .iter()
            .position(|f| f == "--chat-template")
            .expect("has --chat-template");
        assert_eq!(flags[idx + 1], "chatml");
    }

    // ─── Stop string parsing test ───────────────────────────────────────────

    #[test]
    fn test_stop_string_parsing() {
        clear_env();

        // Unset → empty
        let config = Config::from_env();
        assert!(config.stop.is_empty());

        // Comma-separated with trimming
        unsafe { std::env::set_var("LLAMA_STOP", "<|end|>, ###, STOP") };
        let config = Config::from_env();
        assert_eq!(config.stop, vec!["<|end|>", "###", "STOP"]);

        unsafe { std::env::remove_var("LLAMA_STOP") };
    }
}
