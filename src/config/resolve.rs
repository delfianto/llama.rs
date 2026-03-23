use std::path::{Path, PathBuf};

use crate::download::hf::ModelSpec;
use crate::error::LlamaError;

/// Resolve a model path from user input.
///
/// Resolution order:
/// 1. Absolute path (`/path/to/model.gguf`) → use as-is
/// 2. Model spec with colon (`org/repo:quant`) → resolve to `models_dir/org/repo-quant.gguf`
/// 3. Relative path containing `/` → treat as relative under `models_dir`
/// 4. Plain filename → search `models_dir` recursively
pub fn resolve_model_path(models_dir: &Path, input: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(input);

    // 1. Absolute path
    if path.is_absolute() {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
        anyhow::bail!(LlamaError::ModelNotFound {
            path: input.to_string(),
        });
    }

    // 2. Model spec: org/repo:quant
    if input.contains(':') {
        if let Some(spec) = ModelSpec::parse(input) {
            let full = spec.local_path(models_dir);
            if full.is_file() {
                return Ok(full);
            }
            anyhow::bail!(LlamaError::ModelNotFound {
                path: format!(
                    "{}. Run 'llama pull {input}' to download it",
                    spec.display_name()
                ),
            });
        }
    }

    // 3. Relative path (contains slash, no colon)
    if input.contains('/') {
        // Try as-is (with .gguf)
        let full = models_dir.join(input);
        if full.is_file() {
            return Ok(full);
        }
        // Try appending .gguf
        let with_ext = models_dir.join(format!("{input}.gguf"));
        if with_ext.is_file() {
            return Ok(with_ext);
        }
        anyhow::bail!(LlamaError::ModelNotFound {
            path: format!("{input} (looked in {})", models_dir.display()),
        });
    }

    // 4. Plain filename — search recursively
    if let Some(found) = search_recursive(models_dir, input) {
        return Ok(found);
    }

    anyhow::bail!(LlamaError::ModelNotFound {
        path: format!(
            "{input}. Run 'llama ls' to see available models or 'llama pull' to download"
        ),
    });
}

/// Walk `dir` recursively looking for a file with the exact `filename`.
fn search_recursive(dir: &Path, filename: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();

    for entry in entries.filter_map(Result::ok) {
        let ft = entry.file_type().ok()?;
        if ft.is_file() && entry.file_name() == std::ffi::OsStr::new(filename) {
            return Some(entry.path());
        }
        if ft.is_dir() {
            subdirs.push(entry.path());
        }
    }

    for sub in subdirs {
        if let Some(found) = search_recursive(&sub, filename) {
            return Some(found);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Set up a models dir matching the new flat structure:
    /// models_dir/org/repo-quant.gguf
    fn setup_models_dir() -> TempDir {
        let tmp = TempDir::new().expect("create temp dir");

        // org/repo-quant.gguf (new flat structure)
        let org_dir = tmp.path().join("mradermacher");
        fs::create_dir_all(&org_dir).expect("create org dir");
        fs::write(org_dir.join("test-repo-GGUF-Q4_K_M.gguf"), b"fake model").expect("write model");

        // A top-level model (absolute path test)
        fs::write(tmp.path().join("simple.gguf"), b"fake model").expect("write model");

        tmp
    }

    #[test]
    fn test_absolute_path_resolution() {
        let tmp = setup_models_dir();
        let model_path = tmp.path().join("simple.gguf");
        let result = resolve_model_path(tmp.path(), model_path.to_str().expect("valid path"));
        assert!(result.is_ok());
        assert_eq!(result.expect("just checked"), model_path);
    }

    #[test]
    fn test_absolute_path_not_found() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "/nonexistent/model.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_spec_resolution() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "mradermacher/test-repo-GGUF:Q4_K_M");
        assert!(result.is_ok());
        assert_eq!(
            result.expect("just checked"),
            tmp.path().join("mradermacher/test-repo-GGUF-Q4_K_M.gguf")
        );
    }

    #[test]
    fn test_spec_not_found_suggests_pull() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "org/nonexistent:Q4_K_M");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("llama pull"),
            "Error should suggest pull: {err}"
        );
    }

    #[test]
    fn test_relative_path_resolution() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "mradermacher/test-repo-GGUF-Q4_K_M.gguf");
        assert!(result.is_ok());
    }

    #[test]
    fn test_relative_path_without_extension() {
        let tmp = setup_models_dir();
        // Should find the .gguf file even without the extension
        let result = resolve_model_path(tmp.path(), "mradermacher/test-repo-GGUF-Q4_K_M");
        assert!(result.is_ok());
    }

    #[test]
    fn test_relative_path_not_found() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "org/nonexistent.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_filename_search_resolution() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "test-repo-GGUF-Q4_K_M.gguf");
        assert!(result.is_ok());
        assert!(result
            .expect("just checked")
            .to_string_lossy()
            .ends_with("test-repo-GGUF-Q4_K_M.gguf"));
    }

    #[test]
    fn test_filename_search_top_level() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "simple.gguf");
        assert!(result.is_ok());
        assert_eq!(
            result.expect("just checked"),
            tmp.path().join("simple.gguf")
        );
    }

    #[test]
    fn test_model_not_found_error() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "nonexistent.gguf");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent.gguf"));
    }
}
