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

            let quant = extract_quant(file);

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
/// Looks for well-known quantization patterns (Q4_K_M, IQ4_XS, F16, etc.)
/// at the end of the filename, optionally preceded by a prefix like `UD-` or `i1-`.
fn extract_quant(filename: &str) -> String {
    // Find the rightmost position where a quant pattern starts.
    // Quant patterns begin with Q, IQ, F, or BF followed by a digit.
    // They may be preceded by a prefix like "UD-" or "i1-".
    let upper = filename.to_uppercase();
    let bytes = upper.as_bytes();

    // Scan right-to-left for the start of a quant token
    let mut quant_start = None;
    for i in (0..bytes.len()).rev() {
        if is_quant_start(bytes, i) {
            quant_start = Some(i);
            break;
        }
    }

    let Some(start) = quant_start else {
        return String::new();
    };

    // Use the original case from the filename
    let quant = &filename[start..];

    // Check for a known prefix before the quant (e.g., "UD-", "i1-")
    if start >= 3 {
        let prefix_region = &filename[..start];
        // Look for prefixes like "UD-", "i1-" attached via separator
        if let Some(p) = prefix_region.strip_suffix('-') {
            if let Some(prefix) = p.rsplit(['-', '.']).next() {
                let pu = prefix.to_uppercase();
                if pu == "UD" || pu == "I1" {
                    let prefix_start = start - prefix.len() - 1; // -1 for the '-'
                    return filename[prefix_start..].to_string();
                }
            }
        }
    }

    quant.to_string()
}

/// Check if position `i` in the uppercased byte slice starts a quant token.
/// Matches: Q\d, IQ\d, F\d, BF\d
fn is_quant_start(bytes: &[u8], i: usize) -> bool {
    // Must be at start of string or preceded by a separator (-, .)
    if i > 0 && bytes[i - 1] != b'-' && bytes[i - 1] != b'.' {
        return false;
    }

    let remaining = &bytes[i..];
    if remaining.len() < 2 {
        return false;
    }

    match remaining[0] {
        b'Q' | b'F' => remaining[1].is_ascii_digit(),
        b'I' => remaining.len() >= 3 && remaining[1] == b'Q' && remaining[2].is_ascii_digit(),
        b'B' => remaining.len() >= 3 && remaining[1] == b'F' && remaining[2].is_ascii_digit(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_matching_prefix() {
        let d = parse_model_display(
            "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M",
        );
        assert_eq!(d.org, "HauhauCS");
        assert_eq!(d.model, "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive");
        assert_eq!(d.quant, "Q4_K_M");
    }

    #[test]
    fn test_parse_ud_prefix_quant() {
        let d = parse_model_display("unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-UD-Q4_K_XL");
        assert_eq!(d.org, "unsloth");
        assert_eq!(d.model, "gemma-4-E2B-it-GGUF");
        assert_eq!(d.quant, "UD-Q4_K_XL");
    }

    #[test]
    fn test_parse_dot_separator_quant() {
        // mradermacher style: model.Q8_0
        let d = parse_model_display(
            "mradermacher/Gemma-3-Prompt-Coder-270m-it-Uncensored-GGUF/Gemma-3-Prompt-Coder-270m-it-Uncensored.Q8_0",
        );
        assert_eq!(d.quant, "Q8_0");
    }

    #[test]
    fn test_parse_dot_separator_i1_quant() {
        // mradermacher i1 style: model.i1-Q4_K_M
        let d = parse_model_display(
            "mradermacher/gpt-oss-20b-heretic-ara-v3-i1-GGUF/gpt-oss-20b-heretic-ara-v3.i1-Q4_K_M",
        );
        assert_eq!(d.quant, "i1-Q4_K_M");
    }

    #[test]
    fn test_parse_mismatched_filename() {
        // TheDrummer style: repo=v1-GGUF but file=v1b-Q4_K_M
        let d =
            parse_model_display("TheDrummer/Rocinante-X-12B-v1-GGUF/Rocinante-X-12B-v1b-Q4_K_M");
        assert_eq!(d.quant, "Q4_K_M");
    }

    #[test]
    fn test_parse_lowercase_quant() {
        // TeichAI style: model.q4_k_m
        let d = parse_model_display(
            "TeichAI/gemma-4-31B-it-Claude-Opus-Distill-GGUF/gemma-4-31B-it-Claude-Opus-Distill.q4_k_m",
        );
        assert_eq!(d.quant, "q4_k_m");
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
