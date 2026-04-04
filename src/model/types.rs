use std::path::PathBuf;
use std::time::SystemTime;

/// Metadata for a discovered GGUF model file.
pub struct ModelInfo {
    /// Display name relative to models_dir (e.g., `org/repo/model.gguf`).
    pub name: String,
    /// Absolute path on disk.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Last modified time.
    pub modified: SystemTime,
}

/// Parsed display components for a model path.
pub struct ModelDisplay {
    /// Organization / uploader (e.g., "HauhauCS", "unsloth")
    pub org: String,
    /// Model/repo name (e.g., "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive")
    pub model: String,
    /// Quantization format (e.g., "Q4_K_M", "F16")
    pub quant: String,
}

/// Parse a model display name (org/repo/file without .gguf) into structured components.
///
/// For 3-level paths like `HauhauCS/Repo-Name/Repo-Name-Q4_K_M`:
/// - Extracts org, repo, and derives quant by stripping the repo name prefix from the filename.
///
/// For simpler paths, does best-effort extraction.
pub fn parse_model_display(name: &str) -> ModelDisplay {
    let parts: Vec<&str> = name.splitn(3, '/').collect();

    match parts.len() {
        3 => {
            let org = parts[0].to_string();
            let repo = parts[1].to_string();
            let file = parts[2];

            // Try to extract quant by stripping repo name prefix from filename
            let quant = extract_quant(file, &repo);

            ModelDisplay {
                org,
                model: repo,
                quant,
            }
        }
        2 => {
            let org = parts[0].to_string();
            let file = parts[1];
            ModelDisplay {
                org,
                model: file.to_string(),
                quant: String::new(),
            }
        }
        _ => ModelDisplay {
            org: String::new(),
            model: name.to_string(),
            quant: String::new(),
        },
    }
}

/// Extract the quantization format from a filename given the repo name.
///
/// Strategy:
/// 1. Strip the repo name (minus trailing `-GGUF`) as a prefix from the filename
/// 2. Whatever remains after a leading `-` is the quant
/// 3. Fallback: extract the last segment that looks like a quant pattern
fn extract_quant(filename: &str, repo: &str) -> String {
    // Strip common suffixes from repo name for prefix matching
    let repo_base = repo
        .strip_suffix("-GGUF")
        .or_else(|| repo.strip_suffix("-gguf"))
        .unwrap_or(repo);

    // Try stripping repo_base as prefix from the filename
    if let Some(remainder) = filename.strip_prefix(repo_base) {
        let quant = remainder.trim_start_matches('-');
        if !quant.is_empty() {
            return quant.to_string();
        }
    }

    // Fallback: take the last hyphen-separated segment(s) that look like a quant
    // Common patterns: Q4_K_M, Q8_0, IQ4_XS, F16, BF16, UD-Q4_K_XL
    filename
        .rsplit_once('-')
        .map(|(_, q)| q.to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_3_level_matching_prefix() {
        let d = parse_model_display(
            "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M",
        );
        assert_eq!(d.org, "HauhauCS");
        assert_eq!(d.model, "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive");
        assert_eq!(d.quant, "Q4_K_M");
    }

    #[test]
    fn test_parse_3_level_gguf_suffix_repo() {
        let d = parse_model_display("unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-UD-Q4_K_XL");
        assert_eq!(d.org, "unsloth");
        assert_eq!(d.model, "gemma-4-E2B-it-GGUF");
        assert_eq!(d.quant, "UD-Q4_K_XL");
    }

    #[test]
    fn test_parse_2_level() {
        let d = parse_model_display("org/model-file");
        assert_eq!(d.org, "org");
        assert_eq!(d.model, "model-file");
        assert_eq!(d.quant, "");
    }

    #[test]
    fn test_parse_1_level() {
        let d = parse_model_display("simple");
        assert_eq!(d.org, "");
        assert_eq!(d.model, "simple");
        assert_eq!(d.quant, "");
    }
}
