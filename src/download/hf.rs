use std::path::{Path, PathBuf};

use serde::Deserialize;

/// A parsed model spec in `[hf.co/]org/repo:quant` format.
///
/// Examples:
/// - `mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M`
/// - `hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M`
pub struct ModelSpec {
    /// The org (e.g., `mradermacher`)
    pub org: String,
    /// The repo name (e.g., `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF`)
    pub repo: String,
    /// The quantization tag (e.g., `Q4_K_M`)
    pub quant: String,
}

impl ModelSpec {
    /// Parse a model spec string.
    ///
    /// Accepts:
    /// - `org/repo:quant`
    /// - `hf.co/org/repo:quant` (ollama-style, strips `hf.co/` prefix)
    pub fn parse(spec: &str) -> Option<Self> {
        // Strip hf.co/ prefix if present
        let spec = spec.strip_prefix("hf.co/").unwrap_or(spec);

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

    /// The HuggingFace API URL to list repo contents.
    pub fn api_url(&self) -> String {
        format!(
            "https://huggingface.co/api/models/{}/{}",
            self.org, self.repo
        )
    }

    /// Build the download URL for a specific filename.
    pub fn download_url(&self, filename: &str) -> String {
        format!(
            "https://huggingface.co/{}/{}/resolve/main/{}",
            self.org, self.repo, filename
        )
    }

    /// The local repo directory: `{models_dir}/{org}/{repo}/`
    pub fn local_dir(&self, models_dir: &Path) -> PathBuf {
        models_dir.join(&self.org).join(&self.repo)
    }

    /// The local file path: `{models_dir}/{org}/{repo}/{gguf_filename}`
    pub fn local_path(&self, models_dir: &Path, gguf_filename: &str) -> PathBuf {
        self.local_dir(models_dir).join(gguf_filename)
    }

    /// The display name: `{org}/{repo}:{quant}` (echoes the spec format).
    pub fn display_name(&self) -> String {
        format!("{}/{}:{}", self.org, self.repo, self.quant)
    }

    /// The HuggingFace repo id: `{org}/{repo}`
    pub fn repo_id(&self) -> String {
        format!("{}/{}", self.org, self.repo)
    }
}

/// Metadata files to download alongside the GGUF model (LM Studio pattern).
const METADATA_FILES: &[&str] = &[
    "README.md",
    "config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// Result of resolving a model spec against the HuggingFace API.
#[derive(Debug)]
pub struct ResolvedModel {
    /// The actual GGUF filename in the repo.
    pub gguf_filename: String,
    /// Metadata files present in the repo that should be downloaded alongside.
    pub metadata_files: Vec<String>,
}

/// Response from `GET https://huggingface.co/api/models/{org}/{repo}`.
#[derive(Deserialize)]
pub struct HfModelInfo {
    pub siblings: Option<Vec<HfSibling>>,
}

/// A file entry in the HuggingFace model info response.
#[derive(Deserialize)]
pub struct HfSibling {
    pub rfilename: String,
}

/// Resolve the actual GGUF filename and available metadata files in a HuggingFace repo.
///
/// Calls `GET https://huggingface.co/api/models/{org}/{repo}` to list files,
/// then finds the `.gguf` file whose name contains the quant string (case-insensitive),
/// and identifies metadata files (config.json, tokenizer files, etc.) to download alongside.
pub async fn resolve_gguf_filename(
    client: &reqwest::Client,
    spec: &ModelSpec,
) -> anyhow::Result<ResolvedModel> {
    let api_url = spec.api_url();
    let resp = client.get(&api_url).send().await?;

    match resp.status().as_u16() {
        404 => anyhow::bail!(
            "Repository not found: {}. Check the org and repo name.",
            spec.repo_id()
        ),
        401 | 403 => anyhow::bail!(
            "Access denied to {}. Set HF_TOKEN for gated models.",
            spec.repo_id()
        ),
        s if s >= 400 => anyhow::bail!("HuggingFace API error: HTTP {s}"),
        _ => {}
    }

    let info: HfModelInfo = resp.json().await?;

    let siblings = info
        .siblings
        .ok_or_else(|| anyhow::anyhow!("No file listing returned for {}", spec.repo_id()))?;

    // Find .gguf files matching the quant tag (case-insensitive)
    let quant_lower = spec.quant.to_lowercase();
    let mut candidates: Vec<&str> = siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|f| {
            std::path::Path::new(f)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .filter(|f| f.to_lowercase().contains(&quant_lower))
        .collect();

    let gguf_filename = match candidates.len() {
        0 => {
            // List available quants for a helpful error
            let available: Vec<&str> = siblings
                .iter()
                .map(|s| s.rfilename.as_str())
                .filter(|f| {
                    std::path::Path::new(f)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                })
                .collect();

            if available.is_empty() {
                anyhow::bail!("No GGUF files found in {}", spec.repo_id());
            }
            anyhow::bail!(
                "No GGUF file matching '{}' in {}.\nAvailable files:\n  {}",
                spec.quant,
                spec.repo_id(),
                available.join("\n  ")
            );
        }
        1 => candidates[0].to_string(),
        _ => {
            // Multiple matches — pick the one that matches most precisely.
            // Sort by length (shorter = more precise match) and pick first.
            candidates.sort_by_key(|f| f.len());
            candidates[0].to_string()
        }
    };

    // Find metadata files present in the repo
    let repo_files: std::collections::HashSet<&str> =
        siblings.iter().map(|s| s.rfilename.as_str()).collect();
    let metadata_files = METADATA_FILES
        .iter()
        .filter(|f| repo_files.contains(**f))
        .map(|f| (*f).to_string())
        .collect();

    Ok(ResolvedModel {
        gguf_filename,
        metadata_files,
    })
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
    fn test_parse_spec_with_hf_prefix() {
        let spec = ModelSpec::parse("hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M").unwrap();
        assert_eq!(spec.org, "TheDrummer");
        assert_eq!(spec.repo, "Cydonia-24B-v4.3-GGUF");
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
    fn test_api_url() {
        let spec = ModelSpec::parse("TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M").unwrap();
        assert_eq!(
            spec.api_url(),
            "https://huggingface.co/api/models/TheDrummer/Cydonia-24B-v4.3-GGUF"
        );
    }

    #[test]
    fn test_download_url() {
        let spec = ModelSpec::parse("TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M").unwrap();
        assert_eq!(
            spec.download_url("Cydonia-24B-v4zg-Q4_K_M.gguf"),
            "https://huggingface.co/TheDrummer/Cydonia-24B-v4.3-GGUF/resolve/main/Cydonia-24B-v4zg-Q4_K_M.gguf"
        );
    }

    #[test]
    fn test_local_dir() {
        let spec = ModelSpec::parse(
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M",
        )
        .unwrap();
        assert_eq!(
            spec.local_dir(Path::new("/models")),
            Path::new(
                "/models/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF"
            )
        );
    }

    #[test]
    fn test_local_path() {
        let spec = ModelSpec::parse(
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M",
        )
        .unwrap();
        assert_eq!(
            spec.local_path(Path::new("/models"), "Qwen3.5-27B-Q4_K_M.gguf"),
            Path::new(
                "/models/mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
            )
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
            "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M"
        );
    }

    #[test]
    fn test_display_name_from_path() {
        assert_eq!(
            display_name_from_path("mradermacher/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"),
            "mradermacher/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M"
        );
    }

    #[test]
    fn test_display_name_from_path_no_extension() {
        assert_eq!(display_name_from_path("org/repo/model"), "org/repo/model");
    }

    #[test]
    fn test_mozilla_test_model() {
        let spec = ModelSpec::parse("Mozilla/llama-test-model:tiny-llama").unwrap();
        assert_eq!(
            spec.download_url("tiny-llama.gguf"),
            "https://huggingface.co/Mozilla/llama-test-model/resolve/main/tiny-llama.gguf"
        );
        assert_eq!(
            spec.local_dir(Path::new("/models")),
            Path::new("/models/Mozilla/llama-test-model")
        );
        assert_eq!(
            spec.local_path(Path::new("/models"), "tiny-llama.gguf"),
            Path::new("/models/Mozilla/llama-test-model/tiny-llama.gguf")
        );
        assert_eq!(spec.display_name(), "Mozilla/llama-test-model:tiny-llama");
    }

    #[test]
    fn test_hf_prefix_stripped_for_all_methods() {
        let spec = ModelSpec::parse("hf.co/org/repo:Q4_K_M").unwrap();
        assert_eq!(spec.org, "org");
        assert_eq!(spec.repo, "repo");
        assert_eq!(spec.api_url(), "https://huggingface.co/api/models/org/repo");
    }
}
