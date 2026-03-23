# Step 02: Configuration System

## Objective
Implement the `config/` module: load all env vars with defaults into an immutable `Config` struct. This is the foundation everything else reads from.

## Instructions

### 1. Config Struct (`config/mod.rs`)

```rust
pub struct Config {
    // Paths
    pub bin_dir: Option<PathBuf>,     // LLAMA_BIN_DIR, None = search PATH
    pub models_dir: PathBuf,          // LLAMA_MODELS_DIR

    // GPU / Hardware
    pub gpu_layers: u32,              // LLAMA_GPU_LAYERS
    pub tensor_split: Option<String>, // LLAMA_TENSOR_SPLIT
    pub main_gpu: u32,                // LLAMA_MAIN_GPU
    pub flash_attn: bool,             // LLAMA_FLASH_ATTN
    pub mlock: bool,                  // LLAMA_MLOCK

    // Inference
    pub ctx_size: u32,                // LLAMA_CTX_SIZE
    pub batch_size: u32,              // LLAMA_BATCH_SIZE
    pub threads: u32,                 // LLAMA_THREADS

    // Server
    pub host: String,                 // LLAMA_HOST
    pub port: u16,                    // LLAMA_PORT

    // REPL
    pub system_prompt: String,        // LLAMA_SYSTEM_PROMPT

    // Download
    pub download_connections: u8,     // LLAMA_DOWNLOAD_CONNECTIONS
    pub hf_token: Option<String>,     // HF_TOKEN

    // Logging
    pub log_level: String,            // LLAMA_LOG
}
```

### 2. Loading Logic

`Config::from_env()` reads each var with defaults from `CONFIG_REFERENCE.md`. Use `std::env::var()` — no external config crate needed.

For `LLAMA_THREADS` default: use `std::thread::available_parallelism()`.
For `LLAMA_MODELS_DIR` default: use `dirs::data_dir()` / `llama/models` or fallback to `~/.local/share/llama/models`.

### 3. Binary Resolution

`Config::find_binary(&self, name: &str) -> Result<PathBuf>`:
- If `bin_dir` is set, look there
- Otherwise, use `which::which(name)` to search PATH
- Return error with helpful message if not found

Add `which = "7"` to dependencies.

### 4. Model Path Resolution (`config/resolve.rs`)

`resolve_model_path(models_dir: &Path, input: &str) -> Result<PathBuf>`:
1. If `input` starts with `/` → return as-is, check exists
2. If `input` contains `/` → join with `models_dir`, check exists
3. Otherwise → walk `models_dir` recursively for a file matching `input`
4. Error if not found, with suggestion to run `llama pull`

### 5. Common Flags Builder

`Config::build_common_flags(&self, model_path: &Path) -> Vec<String>`:
Produces the CLI arguments for llama-server/llama-cli, mirroring the shell script's `build_common_flags()`.

## Tests

```rust
#[cfg(test)]
mod tests {
    // Test default values
    fn test_defaults_are_sensible() { ... }

    // Test env var override
    fn test_env_override() {
        // Use temp_env or set vars in test
        std::env::set_var("LLAMA_CTX_SIZE", "8192");
        let config = Config::from_env();
        assert_eq!(config.ctx_size, 8192);
    }

    // Test model path resolution
    fn test_absolute_path_resolution() { ... }
    fn test_relative_path_resolution() { ... }
    fn test_filename_search_resolution() { ... }
    fn test_model_not_found_error() { ... }

    // Test flag building
    fn test_common_flags_basic() { ... }
    fn test_common_flags_with_tensor_split() { ... }
    fn test_common_flags_flash_attn_off() { ... }
    fn test_common_flags_mlock_off() { ... }

    // Test binary finding
    fn test_find_binary_with_bin_dir() { ... }
    fn test_find_binary_in_path() { ... }
}
```

## Acceptance Criteria

- [ ] `Config::from_env()` loads all env vars with correct defaults
- [ ] Model path resolution handles all 3 input formats
- [ ] `build_common_flags()` output matches the shell script's flag construction
- [ ] Binary resolution works with both explicit dir and PATH search
- [ ] All tests pass
- [ ] Quality gate passes
