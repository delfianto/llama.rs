use std::path::Path;
use std::process::Command;

use tracing::debug;

use super::StdProcessHandle;
use crate::config::Config;

/// Spawn llama-cli as an interactive REPL with inherited stdio.
///
/// Uses `std::process::Command` (not tokio) so the child owns the terminal
/// for interactive input/output.
pub fn spawn_cli(config: &Config, model_path: &Path) -> anyhow::Result<StdProcessHandle> {
    let binary = config.find_binary("llama-cli")?;
    let mut flags = config.build_common_flags(model_path);

    // REPL-specific flags
    flags.push("--conversation".to_string());
    flags.push("-p".to_string());
    flags.push(config.system_prompt.clone());
    flags.push("--color".to_string());

    debug!("Spawning: {} {}", binary.display(), flags.join(" "));

    let child = Command::new(&binary)
        .args(&flags)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()?;

    let pid = child.id();

    Ok(StdProcessHandle { child, pid })
}

/// Build the full argument list that `spawn_cli` would use (for testing).
pub fn build_cli_args(config: &Config, model_path: &Path) -> Vec<String> {
    let mut flags = config.build_common_flags(model_path);
    flags.push("--conversation".to_string());
    flags.push("-p".to_string());
    flags.push(config.system_prompt.clone());
    flags.push("--color".to_string());
    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn test_config() -> Config {
        // Clear env to get predictable defaults
        for key in [
            "LLAMA_BIN_DIR",
            "LLAMA_TENSOR_SPLIT",
            "LLAMA_FLASH_ATTN",
            "LLAMA_MLOCK",
        ] {
            std::env::remove_var(key);
        }

        let mut config = Config::from_env();
        config.gpu_layers = 999;
        config.ctx_size = 32768;
        config.batch_size = 2048;
        config.threads = 8;
        config.system_prompt = "You are a helpful assistant.".to_string();
        config.tensor_split = None;
        config.flash_attn = true;
        config.mlock = true;
        config
    }

    #[test]
    fn test_cli_flags_contain_conversation() {
        let config = test_config();
        let flags = build_cli_args(&config, Path::new("/models/test.gguf"));
        assert!(flags.contains(&"--conversation".to_string()));
    }

    #[test]
    fn test_cli_flags_contain_system_prompt() {
        let config = test_config();
        let flags = build_cli_args(&config, Path::new("/models/test.gguf"));
        let p_idx = flags.iter().position(|f| f == "-p");
        assert!(p_idx.is_some());
        assert_eq!(
            flags[p_idx.expect("just checked") + 1],
            "You are a helpful assistant."
        );
    }

    #[test]
    fn test_cli_flags_contain_color() {
        let config = test_config();
        let flags = build_cli_args(&config, Path::new("/models/test.gguf"));
        assert!(flags.contains(&"--color".to_string()));
    }

    #[test]
    fn test_cli_flags_contain_model() {
        let config = test_config();
        let flags = build_cli_args(&config, Path::new("/models/test.gguf"));
        let m_idx = flags.iter().position(|f| f == "-m");
        assert!(m_idx.is_some());
        assert_eq!(flags[m_idx.expect("just checked") + 1], "/models/test.gguf");
    }

    #[test]
    fn test_cli_flags_order() {
        let config = test_config();
        let flags = build_cli_args(&config, Path::new("/models/test.gguf"));

        // Common flags come first, then REPL-specific flags
        let m_idx = flags.iter().position(|f| f == "-m").expect("has -m");
        let conv_idx = flags
            .iter()
            .position(|f| f == "--conversation")
            .expect("has --conversation");
        assert!(
            m_idx < conv_idx,
            "common flags should come before REPL flags"
        );
    }
}
