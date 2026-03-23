pub mod types;

use std::path::Path;
use std::time::{Duration, SystemTime};

use types::ModelInfo;

/// Recursively scan `models_dir` for `.gguf` files, returning them sorted by name.
pub fn scan_models(models_dir: &Path) -> anyhow::Result<Vec<ModelInfo>> {
    let mut models = Vec::new();

    if !models_dir.is_dir() {
        return Ok(models);
    }

    scan_recursive(models_dir, models_dir, &mut models);
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

fn scan_recursive(base: &Path, dir: &Path, results: &mut Vec<ModelInfo>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            scan_recursive(base, &path, results);
        } else if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            let relative = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            let name = crate::download::hf::display_name_from_path(&relative);

            let metadata = std::fs::metadata(&path);
            let (size, modified) = match metadata {
                Ok(m) => (m.len(), m.modified().unwrap_or(SystemTime::UNIX_EPOCH)),
                Err(_) => (0, SystemTime::UNIX_EPOCH),
            };

            results.push(ModelInfo {
                name,
                path,
                size,
                modified,
            });
        }
    }
}

/// Find a running process that has `model_path` in its command-line arguments.
///
/// Uses `ps aux` and greps for the model filename. Returns the PID if found.
pub fn find_process_using_model(model_path: &Path) -> anyhow::Result<Option<u32>> {
    let filename = model_path.to_string_lossy().to_string();

    let output = std::process::Command::new("ps").args(["aux"]).output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        // Skip our own process and grep processes
        if !line.contains("llama-server") {
            continue;
        }
        if !line.contains(&filename) {
            continue;
        }

        // Parse PID from ps output (USER PID ...)
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(pid) = parts[1].parse::<u32>() {
                return Ok(Some(pid));
            }
        }
    }

    Ok(None)
}

/// Remove empty parent directories up to (but not including) `stop_at`.
pub fn cleanup_empty_dirs(file_path: &Path, stop_at: &Path) -> anyhow::Result<()> {
    let mut dir = file_path.parent();

    while let Some(d) = dir {
        if d == stop_at {
            break;
        }

        // Try to remove — will only succeed if empty
        match std::fs::remove_dir(d) {
            Ok(()) => {
                dir = d.parent();
            }
            Err(_) => break, // Not empty or other error — stop
        }
    }

    Ok(())
}

/// Format a byte count as human-readable (e.g., "4.1 GB").
pub fn format_size(bytes: u64) -> String {
    bytesize::ByteSize(bytes).to_string()
}

/// Format a `SystemTime` as a relative time string (e.g., "2 days ago").
pub fn format_relative_time(time: SystemTime) -> String {
    let elapsed = time.elapsed().unwrap_or(Duration::ZERO);
    let secs = elapsed.as_secs();

    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        let m = secs / 60;
        format!("{m} min ago")
    } else if secs < 86400 {
        let h = secs / 3600;
        format!("{h} hour{} ago", if h == 1 { "" } else { "s" })
    } else if secs < 2_592_000 {
        let d = secs / 86400;
        format!("{d} day{} ago", if d == 1 { "" } else { "s" })
    } else if secs < 31_536_000 {
        let m = secs / 2_592_000;
        format!("{m} month{} ago", if m == 1 { "" } else { "s" })
    } else {
        let y = secs / 31_536_000;
        format!("{y} year{} ago", if y == 1 { "" } else { "s" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_models() -> TempDir {
        let tmp = TempDir::new().unwrap();

        // New flat structure: org/repo-quant.gguf
        let org1 = tmp.path().join("mradermacher");
        fs::create_dir_all(&org1).unwrap();
        fs::write(org1.join("test-repo-GGUF-Q4_K_M.gguf"), vec![0u8; 1024]).unwrap();

        let org2 = tmp.path().join("TheBloke");
        fs::create_dir_all(&org2).unwrap();
        fs::write(org2.join("Mistral-7B-GGUF-Q4_K_M.gguf"), vec![0u8; 2048]).unwrap();

        // top-level model (e.g., manually placed)
        fs::write(tmp.path().join("simple.gguf"), vec![0u8; 512]).unwrap();

        // non-gguf file (should be ignored)
        fs::write(tmp.path().join("readme.txt"), b"hello").unwrap();

        tmp
    }

    #[test]
    fn test_scan_models_finds_gguf_files() {
        let tmp = setup_models();
        let models = scan_models(tmp.path()).unwrap();
        assert_eq!(models.len(), 3);
    }

    #[test]
    fn test_scan_models_ignores_non_gguf() {
        let tmp = setup_models();
        let models = scan_models(tmp.path()).unwrap();
        assert!(!models.iter().any(|m| m.name.contains("readme")));
    }

    #[test]
    fn test_scan_models_sorted_by_name() {
        let tmp = setup_models();
        let models = scan_models(tmp.path()).unwrap();
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_scan_models_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let models = scan_models(tmp.path()).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_scan_models_nonexistent_dir() {
        let models = scan_models(Path::new("/nonexistent/path")).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_scan_models_has_correct_sizes() {
        let tmp = setup_models();
        let models = scan_models(tmp.path()).unwrap();
        let simple = models.iter().find(|m| m.name == "simple").unwrap();
        assert_eq!(simple.size, 512);
    }

    #[test]
    fn test_cleanup_empty_dirs() {
        let tmp = TempDir::new().unwrap();
        let deep = tmp.path().join("a").join("b").join("c");
        fs::create_dir_all(&deep).unwrap();
        let file = deep.join("model.gguf");
        fs::write(&file, b"data").unwrap();

        // Remove the file, then clean up empty dirs
        fs::remove_file(&file).unwrap();
        cleanup_empty_dirs(&file, tmp.path()).unwrap();

        // All empty dirs should be removed
        assert!(!tmp.path().join("a").exists());
    }

    #[test]
    fn test_cleanup_empty_dirs_stops_at_nonempty() {
        let tmp = TempDir::new().unwrap();
        let deep = tmp.path().join("a").join("b");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("model.gguf"), b"data").unwrap();
        fs::write(tmp.path().join("a").join("other.gguf"), b"keep").unwrap();

        fs::remove_file(deep.join("model.gguf")).unwrap();
        cleanup_empty_dirs(&deep.join("model.gguf"), tmp.path()).unwrap();

        // "a" should still exist because it has "other.gguf"
        assert!(tmp.path().join("a").exists());
        // "b" should be removed
        assert!(!deep.exists());
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1024), "1.0 KB");
        // bytesize uses SI-like formatting
        let gb_str = format_size(1_073_741_824);
        assert!(
            gb_str.contains("GB") || gb_str.contains("MB"),
            "Expected GB or MB, got: {gb_str}"
        );
    }

    #[test]
    fn test_format_relative_time_just_now() {
        let time = SystemTime::now();
        assert_eq!(format_relative_time(time), "just now");
    }

    #[test]
    fn test_format_relative_time_days() {
        let time = SystemTime::now() - Duration::from_secs(2 * 86400);
        assert_eq!(format_relative_time(time), "2 days ago");
    }
}
