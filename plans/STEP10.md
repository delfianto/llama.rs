# Step 10: Model Manager — List & Delete

## Objective
Implement `llama ls` (list models) and `llama rm` (delete model with running process check).

## Instructions

### 1. Model Types (`model/types.rs`)

```rust
pub struct ModelInfo {
    pub name: String,           // Display name: "org/repo/filename.gguf" relative to models_dir
    pub path: PathBuf,          // Absolute path
    pub size: u64,              // File size in bytes
    pub modified: SystemTime,   // Last modified time
}
```

### 2. Model Scanner (`model/mod.rs`)

```rust
pub fn scan_models(models_dir: &Path) -> Result<Vec<ModelInfo>> {
    // Walk models_dir recursively
    // Collect all .gguf files
    // Build ModelInfo for each
    // Sort by name (alphabetically)
}
```

Use `std::fs::read_dir` recursively or `walkdir` crate (add to deps if needed — or just use a simple recursive function to avoid a new dep).

### 3. List Display (`cli/ls.rs`)

```rust
pub fn exec_ls(config: &Config) -> Result<()> {
    let models = scan_models(&config.models_dir)?;

    if models.is_empty() {
        println!("No models found in {}", config.models_dir.display());
        println!("Use 'llama pull <repo> <file>' to download a model.");
        return Ok(());
    }

    // Table output:
    // NAME                                                    SIZE      MODIFIED
    // mradermacher/L3.3-70B-Euryale/Q4_K_M.gguf             42.5 GB   2 days ago
    // TheBloke/Mistral-7B/mistral-7b-v0.1.Q4_K_M.gguf        4.1 GB   1 week ago

    for model in &models {
        println!("{:<55} {:>10} {:>12}",
            model.name,
            format_size(model.size),
            format_relative_time(model.modified),
        );
    }
}
```

Use `bytesize` crate for human-readable sizes. Implement simple relative time formatting (or use `chrono`).

### 4. Delete Model (`cli/rm.rs`)

```rust
pub async fn exec_rm(config: &Config, model: &str) -> Result<()> {
    let model_path = resolve_model_path(&config.models_dir, model)?;

    // Check if a llama-server process is using this model
    if let Some(pid) = find_process_using_model(&model_path)? {
        info!("Model is in use by process {} — stopping it", pid);
        kill_process(pid)?;
        // Brief wait for process to exit
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    // Delete the file
    tokio::fs::remove_file(&model_path).await?;
    info!("Deleted: {}", model_path.display());

    // Clean up empty parent directories
    cleanup_empty_dirs(&model_path, &config.models_dir)?;

    Ok(())
}
```

### 5. Process Detection

`fn find_process_using_model(model_path: &Path) -> Result<Option<u32>>`:

Strategy — check `/proc` on Linux or `ps` on macOS:
- Run `ps aux` and grep for the model filename in process arguments
- Parse out PID if found
- This is simple and avoids heavy dependencies

Alternatively, maintain a PID file when `llama serve` starts:
- Write PID to `$LLAMA_MODELS_DIR/.pids/{model_hash}.pid`
- Check on `llama rm`
- Clean up on graceful shutdown

Use the PID file approach as primary, `ps` scan as fallback.

### 6. Empty Directory Cleanup

After deleting a model, walk up the directory tree (up to `models_dir`) and remove empty directories. This keeps the tree clean after `llama rm`.

## Tests

```rust
#[cfg(test)]
mod tests {
    // Use tempdir with fake .gguf files
    fn test_scan_models_finds_gguf_files() { ... }
    fn test_scan_models_ignores_non_gguf() { ... }
    fn test_scan_models_sorted_by_name() { ... }
    fn test_scan_models_empty_dir() { ... }

    fn test_rm_deletes_file() { ... }
    fn test_rm_cleans_empty_parent_dirs() { ... }
    fn test_rm_nonexistent_errors() { ... }

    fn test_format_size() { ... }
    fn test_format_relative_time() { ... }
}
```

## Acceptance Criteria

- [ ] `llama ls` shows all .gguf files sorted by name with size and date
- [ ] `llama ls` handles empty models directory gracefully
- [ ] `llama rm <model>` deletes the file
- [ ] `llama rm` stops running llama-server using the model before deleting
- [ ] Empty parent directories are cleaned up after deletion
- [ ] Human-readable file sizes and relative timestamps
- [ ] Quality gate passes
