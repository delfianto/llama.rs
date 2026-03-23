# Step 05: API Proxy — OpenAI Passthrough (Non-Streaming)

## Objective
Set up the axum HTTP server and implement non-streaming OpenAI API passthrough. Requests to `/v1/chat/completions` (with `stream: false`) are forwarded to llama-server and the response relayed back.

## Instructions

### 1. Axum Router (`api/mod.rs`)

```rust
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // OpenAI-compatible
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        // Ollama-compatible (stubbed for now)
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/api/tags", get(ollama::tags))
        .route("/api/show", post(ollama::show))
        // Health
        .route("/health", get(health))
        .with_state(state)
}
```

### 2. App State

```rust
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub llama_server_url: String,  // e.g., "http://127.0.0.1:54321"
    pub model_name: String,        // for /v1/models response
    pub http_client: reqwest::Client,
}
```

### 3. OpenAI Passthrough (`api/openai.rs`)

**`chat_completions`** handler:
1. Receive the raw JSON body (don't fully deserialize — passthrough)
2. Check if `stream` field is true → handle differently (Step 06)
3. Forward the request body to `{llama_server_url}/v1/chat/completions`
4. Relay the response body and status code back
5. Copy relevant headers (`content-type`)

**`list_models`** handler:
Return a minimal OpenAI-compatible model list:
```json
{
  "object": "list",
  "data": [{
    "id": "<model_filename>",
    "object": "model",
    "owned_by": "local"
  }]
}
```

### 4. Request/Response Types (`api/types.rs`)

Define minimal types needed for inspection (checking `stream` field) without fully typing the OpenAI spec:
```rust
#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub stream: bool,
    // Capture everything else as raw Value for passthrough
    #[serde(flatten)]
    pub rest: serde_json::Value,
}
```

### 5. Integrate Into `llama serve`

Update `cli/serve.rs` to start the axum server after health check passes:

```rust
let state = AppState { ... };
let router = api::build_router(state);
let listener = tokio::net::TcpListener::bind(format!("{}:{}", config.host, config.port)).await?;
info!("Proxy server listening on http://{}:{}", config.host, config.port);
axum::serve(listener, router)
    .with_graceful_shutdown(shutdown_signal())
    .await?;
```

### 6. CORS

Add permissive CORS via `tower_http::cors::CorsLayer::permissive()` — local development tool, not a production API.

## Tests

```rust
#[cfg(test)]
mod tests {
    // Use wiremock to mock llama-server
    async fn test_chat_completions_passthrough() {
        // Set up wiremock to respond with a canned completion
        // Send request to our proxy
        // Assert response matches
    }

    async fn test_list_models_returns_model_info() { ... }

    async fn test_chat_completions_preserves_status_code() {
        // Mock llama-server returning 400
        // Assert proxy relays the 400
    }

    async fn test_cors_headers_present() { ... }
}
```

## Acceptance Criteria

- [ ] `llama serve` starts axum proxy after llama-server is ready
- [ ] `POST /v1/chat/completions` with `stream: false` returns correct completion
- [ ] `GET /v1/models` returns model list
- [ ] Error responses from llama-server are forwarded correctly
- [ ] CORS headers allow cross-origin requests
- [ ] Ollama endpoints return 501 Not Implemented (placeholder)
- [ ] Quality gate passes
