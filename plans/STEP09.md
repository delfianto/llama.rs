# Step 09: HuggingFace Download Manager

## Objective
Implement `llama pull` to download GGUF model files from HuggingFace with parallel chunk downloads and progress display.

## Instructions

### 1. Download URL Resolution (`download/hf.rs`)

HuggingFace file download URL pattern:
```
https://huggingface.co/{org}/{repo}/resolve/main/{filename}
```

Example:
```
llama pull mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF Q4_K_M.gguf
→ https://huggingface.co/mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF/resolve/main/Q4_K_M.gguf
```

```rust
pub fn resolve_hf_url(repo: &str, filename: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, filename)
}
```

### 2. Parallel Chunk Download (`download/mod.rs`)

```rust
pub async fn download_file(
    url: &str,
    dest: &Path,
    connections: u8,
    hf_token: Option<&str>,
    progress: &ProgressHandler,
) -> Result<()> {
    // 1. HEAD request to get Content-Length and check Accept-Ranges
    // 2. If server supports range requests and file > threshold (e.g., 100MB):
    //    - Split into N chunks based on connection count
    //    - Download each chunk concurrently with Range headers
    //    - Write to temp file at correct offsets (or use separate temp files + concat)
    // 3. If no range support:
    //    - Fall back to single-stream download
    // 4. On completion, rename temp file to final destination
    // 5. On error/interrupt, clean up temp files
}
```

**Chunk download strategy**:
- Use `tokio::task::JoinSet` to run N downloads concurrently
- Each task downloads a byte range: `Range: bytes={start}-{end}`
- Write each chunk to `{dest}.part{N}` temp file
- After all complete, concatenate into final file
- Include HF token as `Authorization: Bearer {token}` header if provided

### 3. Local Directory Structure

```rust
pub fn local_model_path(models_dir: &Path, repo: &str, filename: &str) -> PathBuf {
    // repo = "mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF"
    // → models_dir/mradermacher/L3.3-70B-Euryale-v2.3-heretic-i1-GGUF/Q4_K_M.gguf
    models_dir.join(repo).join(filename)
}
```

Create parent directories automatically.

### 4. Progress Display (`download/progress.rs`)

Use `indicatif` for terminal progress:

```rust
pub struct ProgressHandler {
    multi: MultiProgress,
    bars: Vec<ProgressBar>,
}
```

- One overall progress bar showing total downloaded / total size
- Optionally per-chunk progress bars for verbose mode
- Show download speed and ETA
- Clean display even on small terminals

### 5. Wire Up `llama pull` (`cli/pull.rs`)

```rust
pub async fn exec_pull(config: &Config, repo: &str, filename: &str) -> Result<()> {
    let url = resolve_hf_url(repo, filename);
    let dest = local_model_path(&config.models_dir, repo, filename);

    if dest.exists() {
        info!("Model already exists: {}", dest.display());
        // Ask to re-download? Or just skip.
        return Ok(());
    }

    // Create parent dirs
    tokio::fs::create_dir_all(dest.parent().unwrap()).await?;

    info!("Downloading {} → {}", url, dest.display());
    let progress = ProgressHandler::new();
    download_file(&url, &dest, config.download_connections, config.hf_token.as_deref(), &progress).await?;
    info!("Download complete: {}", dest.display());
    Ok(())
}
```

### 6. Error Handling

- **404**: "Model not found on HuggingFace. Check the repo name and filename."
- **401/403**: "Access denied. This may be a gated model — set HF_TOKEN env var."
- **Network errors**: Retry individual chunks (up to 3 retries with backoff)
- **Interrupted download**: Clean up partial files. Future improvement: resume support.
- **Disk full**: Detect and report clearly.

## Tests

```rust
#[cfg(test)]
mod tests {
    fn test_resolve_hf_url() { ... }
    fn test_local_model_path() { ... }

    // Use wiremock to simulate HF download
    async fn test_single_stream_download() { ... }
    async fn test_parallel_chunk_download() { ... }
    async fn test_download_with_auth_token() { ... }
    async fn test_download_404_error() { ... }
    async fn test_download_creates_directory_structure() { ... }
    async fn test_skip_existing_file() { ... }
}
```

## Acceptance Criteria

- [ ] `llama pull org/repo filename.gguf` downloads the file
- [ ] Files are stored at `$LLAMA_MODELS_DIR/org/repo/filename.gguf`
- [ ] Parallel downloads work when server supports Range requests
- [ ] Falls back to single-stream when Range not supported
- [ ] Progress bar shows speed and ETA
- [ ] HF_TOKEN is sent for gated models
- [ ] Existing files are not re-downloaded
- [ ] Temp files are cleaned up on error
- [ ] Quality gate passes
