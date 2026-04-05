use std::path::{Path, PathBuf};

use crate::download::hf::ModelSpec;
use crate::error::LlamaError;
use crate::model::types::parse_model_display;

/// Resolve a model path from user input.
///
/// Resolution order:
/// 1. Absolute path (`/path/to/model.gguf`) → use as-is
/// 2. Model spec with colon (`org/repo:quant`) → resolve to `models_dir/org/repo/file.gguf`
/// 3. Relative path containing `/` → treat as relative under `models_dir`
/// 4. Directory name match → find repo dir by name, pick GGUF (prompt if multiple)
/// 5. Plain filename → search `models_dir` recursively
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
            let repo_dir = spec.local_dir(models_dir);
            if repo_dir.is_dir() {
                if let Some(found) = find_gguf_by_quant(&repo_dir, &spec.quant) {
                    return Ok(found);
                }
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

    // 4. Directory name match — find a repo directory matching the input
    if let Some(found) = find_by_directory_name(models_dir, input)? {
        return Ok(found);
    }

    // 5. Plain filename — search recursively
    if let Some(found) = search_recursive(models_dir, input) {
        return Ok(found);
    }

    anyhow::bail!(LlamaError::ModelNotFound {
        path: format!(
            "{input}. Run 'llama ls' to see available models or 'llama pull' to download"
        ),
    });
}

/// Search for a repo directory whose name matches `name` (case-insensitive).
///
/// Walks `models_dir/org/repo/` looking for a repo dir that matches.
/// If found, collects model GGUF files in that directory:
/// - 0 files → returns None
/// - 1 file → returns it directly
/// - N files → prompts user to pick one
fn find_by_directory_name(models_dir: &Path, name: &str) -> anyhow::Result<Option<PathBuf>> {
    let name_lower = name.to_lowercase();

    let Ok(orgs) = std::fs::read_dir(models_dir) else {
        return Ok(None);
    };

    // Collect all matching repo directories
    let mut matches: Vec<PathBuf> = Vec::new();

    for org_entry in orgs.filter_map(Result::ok) {
        if !org_entry.path().is_dir() {
            continue;
        }
        let Ok(repos) = std::fs::read_dir(org_entry.path()) else {
            continue;
        };
        for repo_entry in repos.filter_map(Result::ok) {
            let repo_path = repo_entry.path();
            if !repo_path.is_dir() {
                continue;
            }
            if let Some(dir_name) = repo_path.file_name().and_then(|n| n.to_str()) {
                if dir_name.to_lowercase() == name_lower {
                    matches.push(repo_path);
                }
            }
        }
    }

    if matches.is_empty() {
        return Ok(None);
    }

    // Use the first matching directory (there should typically be only one)
    let repo_dir = &matches[0];
    let gguf_files = list_model_ggufs(repo_dir);

    match gguf_files.len() {
        0 => Ok(None),
        1 => Ok(Some(gguf_files.into_iter().next().expect("length checked"))),
        _ => {
            let selected = prompt_select_model(&gguf_files, models_dir)?;
            Ok(Some(selected))
        }
    }
}

/// List GGUF files in a directory, excluding non-model files like mmproj.
fn list_model_ggufs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut ggufs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let path = e.path();
            if !path.is_file() {
                return None;
            }
            let is_gguf = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
            if !is_gguf {
                return None;
            }
            // Skip non-model files (mmproj, etc.)
            let stem = path.file_stem()?.to_str()?;
            if stem.starts_with("mmproj") {
                return None;
            }
            Some(path)
        })
        .collect();

    ggufs.sort();
    ggufs
}

/// Prompt the user to select a model from multiple GGUF files using an interactive list.
fn prompt_select_model(files: &[PathBuf], models_dir: &Path) -> anyhow::Result<PathBuf> {
    let choices: Vec<String> = files
        .iter()
        .map(|f| {
            let relative = f
                .strip_prefix(models_dir)
                .unwrap_or(f)
                .to_string_lossy()
                .to_string();
            let display_name = relative
                .strip_suffix(".gguf")
                .unwrap_or(&relative)
                .to_string();
            let parsed = parse_model_display(&display_name);

            let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
            let size_str = crate::model::format_size(size);

            if parsed.quant.is_empty() {
                format!("{display_name}  ({size_str})")
            } else {
                format!("{}  ({size_str})", parsed.quant)
            }
        })
        .collect();

    println!("Multiple models found. Select one:\n");

    let question = requestty::Question::select("model")
        .message("")
        .choices(choices)
        .build();

    let answer = requestty::prompt_one(question)?;
    let index = answer
        .as_list_item()
        .map(|item| item.index)
        .ok_or_else(|| anyhow::anyhow!("No model selected"))?;

    Ok(files[index].clone())
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

/// Search a directory for a `.gguf` file whose name contains the quant tag (case-insensitive).
fn find_gguf_by_quant(dir: &Path, quant: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let quant_lower = quant.to_lowercase();

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() {
            let is_gguf = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
            if is_gguf {
                let fname = path.file_name()?.to_string_lossy().to_lowercase();
                if fname.contains(&quant_lower) {
                    return Some(path);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Set up a models dir matching the LM Studio 3-level structure:
    /// models_dir/org/repo/file.gguf
    fn setup_models_dir() -> TempDir {
        let tmp = TempDir::new().expect("create temp dir");

        // org/repo/file.gguf (3-level structure)
        let repo_dir = tmp.path().join("mradermacher").join("test-repo-GGUF");
        fs::create_dir_all(&repo_dir).expect("create repo dir");
        fs::write(repo_dir.join("test-repo-GGUF-Q4_K_M.gguf"), b"fake model").expect("write model");

        // A top-level model (absolute path test)
        fs::write(tmp.path().join("simple.gguf"), b"fake model").expect("write model");

        tmp
    }

    /// Set up a models dir with multiple quants in one repo
    fn setup_multi_quant_dir() -> TempDir {
        let tmp = TempDir::new().expect("create temp dir");
        let repo_dir = tmp.path().join("org").join("my-model-GGUF");
        fs::create_dir_all(&repo_dir).expect("create repo dir");
        fs::write(repo_dir.join("my-model-Q4_K_M.gguf"), b"fake").expect("write");
        fs::write(repo_dir.join("my-model-Q8_0.gguf"), b"fake2").expect("write");
        fs::write(repo_dir.join("mmproj-F32.gguf"), b"proj").expect("write mmproj");
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
            tmp.path()
                .join("mradermacher/test-repo-GGUF/test-repo-GGUF-Q4_K_M.gguf")
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
        let result = resolve_model_path(
            tmp.path(),
            "mradermacher/test-repo-GGUF/test-repo-GGUF-Q4_K_M.gguf",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_relative_path_without_extension() {
        let tmp = setup_models_dir();
        // Should find the .gguf file even without the extension
        let result = resolve_model_path(
            tmp.path(),
            "mradermacher/test-repo-GGUF/test-repo-GGUF-Q4_K_M",
        );
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
        assert!(
            result
                .expect("just checked")
                .to_string_lossy()
                .ends_with("test-repo-GGUF-Q4_K_M.gguf")
        );
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

    #[test]
    fn test_directory_name_single_model() {
        let tmp = setup_models_dir();
        // Should find by directory name "test-repo-GGUF"
        let result = resolve_model_path(tmp.path(), "test-repo-GGUF");
        assert!(result.is_ok());
        assert!(
            result
                .expect("just checked")
                .to_string_lossy()
                .ends_with("test-repo-GGUF-Q4_K_M.gguf")
        );
    }

    #[test]
    fn test_directory_name_case_insensitive() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "TEST-REPO-GGUF");
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_model_ggufs_excludes_mmproj() {
        let tmp = setup_multi_quant_dir();
        let repo_dir = tmp.path().join("org").join("my-model-GGUF");
        let ggufs = list_model_ggufs(&repo_dir);
        assert_eq!(ggufs.len(), 2);
        assert!(ggufs.iter().all(|p| {
            !p.file_name()
                .expect("has filename")
                .to_string_lossy()
                .starts_with("mmproj")
        }));
    }

    #[test]
    fn test_directory_name_not_found() {
        let tmp = setup_models_dir();
        let result = resolve_model_path(tmp.path(), "nonexistent-model");
        assert!(result.is_err());
    }
}
