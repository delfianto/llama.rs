# Step 12: Integration Testing & End-to-End Validation

## Objective
Write integration tests that exercise the full binary and API endpoints. Validate compatibility with OpenWebUI and LibreChat request formats.

## Instructions

### 1. CLI Integration Tests (`tests/cli.rs`)

```rust
// Test all subcommands exist and show help
#[test] fn test_run_help() { ... }
#[test] fn test_serve_help() { ... }
#[test] fn test_pull_help() { ... }
#[test] fn test_ls_help() { ... }
#[test] fn test_rm_help() { ... }

// Test model resolution errors
#[test] fn test_run_missing_model_errors() { ... }
#[test] fn test_serve_missing_model_errors() { ... }

// Test env var behavior
#[test] fn test_custom_models_dir() { ... }
#[test] fn test_custom_port() { ... }
```

### 2. API Integration Tests (`tests/api.rs`)

Use a mock llama-server (wiremock) to test the full proxy stack:

```rust
async fn setup_test_server() -> (SocketAddr, MockServer) {
    // Start wiremock as fake llama-server
    // Start our axum proxy pointing at it
    // Return proxy address and mock for assertions
}

// OpenAI endpoint tests
#[tokio::test] async fn test_openai_chat_completion() { ... }
#[tokio::test] async fn test_openai_chat_streaming() { ... }
#[tokio::test] async fn test_openai_list_models() { ... }

// Ollama endpoint tests
#[tokio::test] async fn test_ollama_chat_non_streaming() { ... }
#[tokio::test] async fn test_ollama_chat_streaming_ndjson() { ... }
#[tokio::test] async fn test_ollama_generate() { ... }
#[tokio::test] async fn test_ollama_tags() { ... }
#[tokio::test] async fn test_ollama_show() { ... }

// Compatibility tests — use actual request payloads from frontends
#[tokio::test] async fn test_openwebui_chat_request() { ... }
#[tokio::test] async fn test_librechat_chat_request() { ... }
```

### 3. Download Tests (`tests/download.rs`)

Use wiremock to simulate HuggingFace:

```rust
#[tokio::test] async fn test_pull_creates_directory_structure() { ... }
#[tokio::test] async fn test_pull_parallel_download() { ... }
#[tokio::test] async fn test_pull_auth_header_sent() { ... }
```

### 4. Model Manager Tests (`tests/model.rs`)

Use `tempfile` for isolated model directories:

```rust
#[test] fn test_ls_lists_models_sorted() { ... }
#[test] fn test_rm_deletes_and_cleans_dirs() { ... }
```

### 5. Real llama-server Tests (Optional, `#[ignore]`)

For local development with actual llama.cpp installed:

```rust
#[tokio::test]
#[ignore] // Run with: cargo test -- --ignored
async fn test_real_serve_and_query() {
    // Requires: llama-server in PATH, small test model
    // Start llama serve with tiny model
    // Query /v1/chat/completions
    // Query /api/chat
    // Verify responses
}
```

### 6. Frontend Compatibility Payloads

Capture real request payloads from OpenWebUI and LibreChat and use them as test fixtures to ensure our translation layer handles all fields correctly. Store in `tests/fixtures/`.

## Acceptance Criteria

- [ ] All CLI integration tests pass
- [ ] All API proxy tests pass with mock llama-server
- [ ] OpenAI streaming and non-streaming verified
- [ ] Ollama NDJSON streaming verified
- [ ] OpenWebUI and LibreChat request payloads work
- [ ] Download tests pass with mock HuggingFace
- [ ] Model manager tests pass
- [ ] Quality gate passes
- [ ] `cargo test` runs all non-ignored tests in < 30s
