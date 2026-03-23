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
