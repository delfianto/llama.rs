//! YAML execution profiles for repeatable model experiments.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use super::{ChatTemplate, ComputeDevice, Config};

/// Partial configuration loaded from a YAML execution profile.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExecutionProfile {
    pub model: Option<String>,
    pub paths: PathsProfile,
    pub compute: ComputeProfile,
    pub inference: InferenceProfile,
    pub server: ServerProfile,
    pub prompt: PromptProfile,
    pub sampling: SamplingProfile,
    /// Additional arguments passed directly to both llama-cli and llama-server.
    pub extra_args: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PathsProfile {
    pub bin_dir: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComputeProfile {
    pub device: Option<String>,
    pub gpu_layers: Option<u32>,
    pub tensor_split: Option<String>,
    pub main_gpu: Option<u32>,
    pub flash_attention: Option<bool>,
    pub mlock: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceProfile {
    pub context_size: Option<u32>,
    pub batch_size: Option<u32>,
    pub threads: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerProfile {
    pub host: Option<String>,
    pub port: Option<u16>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PromptProfile {
    pub system: Option<String>,
    pub system_file: Option<PathBuf>,
    pub chat_template: Option<String>,
    pub chat_template_file: Option<PathBuf>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingProfile {
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub context_overflow: Option<String>,
    pub top_k: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
}

impl ExecutionProfile {
    /// Load a profile and return it with the directory relative paths resolve from.
    pub fn load(path: &Path) -> anyhow::Result<(Self, PathBuf)> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        let profile = serde_yaml_ng::from_str(&contents)
            .with_context(|| format!("invalid YAML config {}", path.display()))?;
        let base_dir = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Ok((profile, base_dir))
    }

    /// Overlay this profile onto environment-derived application configuration.
    pub fn apply(self, config: &mut Config, base_dir: &Path) -> anyhow::Result<Option<String>> {
        if let Some(path) = self.paths.bin_dir {
            config.bin_dir = Some(resolve_relative(base_dir, path));
        }
        if let Some(path) = self.paths.models_dir {
            config.models_dir = resolve_relative(base_dir, path);
        }

        if let Some(device) = self.compute.device {
            config.device = device
                .parse::<ComputeDevice>()
                .map_err(|error| anyhow::anyhow!("invalid compute.device: {error}"))?;
        }
        assign(&mut config.gpu_layers, self.compute.gpu_layers);
        if let Some(value) = self.compute.tensor_split {
            config.tensor_split = Some(value);
        }
        if let Some(value) = self.compute.main_gpu {
            config.main_gpu = Some(value);
        }
        assign(&mut config.flash_attn, self.compute.flash_attention);
        assign(&mut config.mlock, self.compute.mlock);

        assign(&mut config.ctx_size, self.inference.context_size);
        assign(&mut config.batch_size, self.inference.batch_size);
        assign(&mut config.threads, self.inference.threads);

        if let Some(value) = self.server.host {
            config.host = value;
        }
        assign(&mut config.port, self.server.port);

        if let Some(path) = self.prompt.system_file {
            config.system_prompt = read_profile_file(base_dir, &path, "prompt.system_file")?;
        } else if let Some(value) = self.prompt.system {
            config.system_prompt = value;
        }
        if let Some(path) = self.prompt.chat_template_file {
            let path = resolve_relative(base_dir, path);
            if !path.is_file() {
                anyhow::bail!(
                    "prompt.chat_template_file does not exist: {}",
                    path.display()
                );
            }
            config.chat_template = Some(ChatTemplate::File(path));
        } else if let Some(value) = self.prompt.chat_template {
            config.chat_template = Some(ChatTemplate::Value(value));
        }
        if let Some(value) = self.prompt.stop {
            config.stop = value;
        }

        assign_some(&mut config.temperature, self.sampling.temperature);
        assign_some(&mut config.max_tokens, self.sampling.max_tokens);
        if let Some(value) = self.sampling.context_overflow {
            if !matches!(value.as_str(), "shift" | "stop") {
                anyhow::bail!("sampling.context_overflow must be 'shift' or 'stop'");
            }
            config.ctx_overflow = value;
        }
        assign_some(&mut config.top_k, self.sampling.top_k);
        assign_some(&mut config.repeat_penalty, self.sampling.repeat_penalty);
        assign_some(&mut config.presence_penalty, self.sampling.presence_penalty);
        assign_some(&mut config.top_p, self.sampling.top_p);
        assign_some(&mut config.min_p, self.sampling.min_p);

        config.extra_args.extend(self.extra_args);
        Ok(self.model)
    }
}

fn assign<T>(target: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *target = value;
    }
}

fn assign_some<T>(target: &mut Option<T>, value: Option<T>) {
    if let Some(value) = value {
        *target = Some(value);
    }
}

fn resolve_relative(base_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn read_profile_file(base_dir: &Path, path: &Path, field: &str) -> anyhow::Result<String> {
    let path = resolve_relative(base_dir, path.to_path_buf());
    std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {field} {}", path.display()))
        .map(|contents| contents.trim().to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_applies_complete_profile() {
        let yaml = r#"
model: org/repo:model-quant
compute:
  device: gpu1
  gpu_layers: 80
  tensor_split: 1,2
  main_gpu: 1
  flash_attention: false
  mlock: false
inference:
  context_size: 65536
  batch_size: 4096
  threads: 12
server:
  host: 0.0.0.0
  port: 9090
prompt:
  system: Be concise.
  stop: ["<end>", "STOP"]
sampling:
  temperature: 0.6
  max_tokens: 1024
  context_overflow: stop
  top_k: 20
  repeat_penalty: 1.1
  presence_penalty: 0.2
  top_p: 0.95
  min_p: 0.05
extra_args:
  - --cache-type-k
  - q8_0
"#;
        let profile: ExecutionProfile = serde_yaml_ng::from_str(yaml).unwrap();
        let mut config = Config::from_env();
        let model = profile.apply(&mut config, Path::new("/profiles")).unwrap();

        assert_eq!(model.as_deref(), Some("org/repo:model-quant"));
        assert_eq!(config.device, ComputeDevice::Devices("CUDA1".to_string()));
        assert_eq!(config.gpu_layers, 80);
        assert_eq!(config.tensor_split.as_deref(), Some("1,2"));
        assert_eq!(config.main_gpu, Some(1));
        assert!(!config.flash_attn);
        assert!(!config.mlock);
        assert_eq!(config.ctx_size, 65536);
        assert_eq!(config.batch_size, 4096);
        assert_eq!(config.threads, 12);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 9090);
        assert_eq!(config.system_prompt, "Be concise.");
        assert_eq!(config.stop, ["<end>", "STOP"]);
        assert_eq!(config.temperature, Some(0.6));
        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(config.ctx_overflow, "stop");
        assert_eq!(config.extra_args, ["--cache-type-k", "q8_0"]);
    }

    #[test]
    fn rejects_unknown_keys() {
        let error = serde_yaml_ng::from_str::<ExecutionProfile>("gpu_layerz: 99")
            .expect_err("unknown field should fail");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn resolves_profile_paths_relative_to_config() {
        let profile: ExecutionProfile =
            serde_yaml_ng::from_str("paths:\n  bin_dir: ../ik/build/bin").unwrap();
        let mut config = Config::from_env();
        profile
            .apply(&mut config, Path::new("/profiles/test"))
            .unwrap();
        assert_eq!(
            config.bin_dir,
            Some(PathBuf::from("/profiles/test/../ik/build/bin"))
        );
    }
}
