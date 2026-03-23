use std::path::{Path, PathBuf};

/// A parsed model spec in `org/repo:quant` format.
///
/// Example: `mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M`
pub struct ModelSpec {
    /// The org (e.g., `mradermacher`)
    pub org: String,
    /// The repo name (e.g., `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF`)
    pub repo: String,
    /// The quantization variant (e.g., `Q4_K_M`)
    pub quant: String,
}

impl ModelSpec {
    /// Parse a model spec string in `org/repo:quant` format.
    ///
    /// Returns `None` if the format is invalid.
    pub fn parse(spec: &str) -> Option<Self> {
        let (org_repo, quant) = spec.split_once(':')?;
        let (org, repo) = org_repo.split_once('/')?;

        if org.is_empty() || repo.is_empty() || quant.is_empty() {
            return None;
        }

        Some(Self {
            org: org.to_string(),
            repo: repo.to_string(),
            quant: quant.to_string(),
        })
    }

    /// The HuggingFace download URL.
    ///
    /// `https://huggingface.co/{org}/{repo}/resolve/main/{quant}.gguf`
    pub fn hf_url(&self) -> String {
        format!(
            "https://huggingface.co/{}/{}/resolve/main/{}.gguf",
            self.org, self.repo, self.quant
        )
    }

    /// The local filename: `{repo}-{quant}.gguf`
    pub fn filename(&self) -> String {
        format!("{}-{}.gguf", self.repo, self.quant)
    }

    /// The local file path: `{models_dir}/{org}/{repo}-{quant}.gguf`
    pub fn local_path(&self, models_dir: &Path) -> PathBuf {
        models_dir.join(&self.org).join(self.filename())
    }

    /// The display name: `{org}/{repo}-{quant}` (no .gguf extension).
    pub fn display_name(&self) -> String {
        format!("{}/{}-{}", self.org, self.repo, self.quant)
    }
}

/// Compute the display name for a model file on disk.
///
/// Given a path relative to models_dir like `org/repo-quant.gguf`,
/// returns `org/repo-quant` (strips `.gguf`).
pub fn display_name_from_path(relative_path: &str) -> String {
    relative_path
        .strip_suffix(".gguf")
        .unwrap_or(relative_path)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_spec() {
        let spec = ModelSpec::parse(
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M",
        )
        .unwrap();
        assert_eq!(spec.org, "mradermacher");
        assert_eq!(
            spec.repo,
            "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF"
        );
        assert_eq!(spec.quant, "Q4_K_M");
    }

    #[test]
    fn test_parse_spec_simple() {
        let spec = ModelSpec::parse("TheBloke/Mistral-7B-GGUF:Q4_K_M").unwrap();
        assert_eq!(spec.org, "TheBloke");
        assert_eq!(spec.repo, "Mistral-7B-GGUF");
        assert_eq!(spec.quant, "Q4_K_M");
    }

    #[test]
    fn test_parse_spec_invalid_no_colon() {
        assert!(ModelSpec::parse("mradermacher/repo").is_none());
    }

    #[test]
    fn test_parse_spec_invalid_no_slash() {
        assert!(ModelSpec::parse("repo:Q4_K_M").is_none());
    }

    #[test]
    fn test_parse_spec_invalid_empty_parts() {
        assert!(ModelSpec::parse("/repo:Q4_K_M").is_none());
        assert!(ModelSpec::parse("org/:Q4_K_M").is_none());
        assert!(ModelSpec::parse("org/repo:").is_none());
    }

    #[test]
    fn test_hf_url() {
        let spec = ModelSpec::parse(
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M",
        )
        .unwrap();
        assert_eq!(
            spec.hf_url(),
            "https://huggingface.co/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF/resolve/main/Q4_K_M.gguf"
        );
    }

    #[test]
    fn test_filename() {
        let spec = ModelSpec::parse(
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M",
        )
        .unwrap();
        assert_eq!(
            spec.filename(),
            "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF-Q4_K_M.gguf"
        );
    }

    #[test]
    fn test_local_path() {
        let spec = ModelSpec::parse(
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M",
        )
        .unwrap();
        assert_eq!(
            spec.local_path(Path::new("/models")),
            Path::new("/models/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF-Q4_K_M.gguf")
        );
    }

    #[test]
    fn test_display_name() {
        let spec = ModelSpec::parse(
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M",
        )
        .unwrap();
        assert_eq!(
            spec.display_name(),
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF-Q4_K_M"
        );
    }

    #[test]
    fn test_display_name_from_path() {
        assert_eq!(
            display_name_from_path("mradermacher/Qwen3.5-27B-GGUF-Q4_K_M.gguf"),
            "mradermacher/Qwen3.5-27B-GGUF-Q4_K_M"
        );
    }

    #[test]
    fn test_display_name_from_path_no_extension() {
        assert_eq!(display_name_from_path("org/model"), "org/model");
    }

    #[test]
    fn test_mozilla_test_model() {
        // Mozilla's test model has a single file, no quant suffix in the filename
        let spec = ModelSpec::parse("Mozilla/llama-test-model:tiny-llama").unwrap();
        assert_eq!(
            spec.hf_url(),
            "https://huggingface.co/Mozilla/llama-test-model/resolve/main/tiny-llama.gguf"
        );
        assert_eq!(spec.filename(), "llama-test-model-tiny-llama.gguf");
        assert_eq!(
            spec.local_path(Path::new("/models")),
            Path::new("/models/Mozilla/llama-test-model-tiny-llama.gguf")
        );
        assert_eq!(spec.display_name(), "Mozilla/llama-test-model-tiny-llama");
    }
}
