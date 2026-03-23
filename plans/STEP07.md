# Step 07: Ollama API — Non-Streaming

## Objective
Implement Ollama-compatible endpoints (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`) with request/response translation. Non-streaming first.

## Instructions

### 1. Ollama Types (`api/types.rs`)

Add Ollama-specific types alongside the OpenAI ones:

```rust
// --- Ollama Request Types ---

#[derive(Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Deserialize, Serialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub num_predict: Option<i32>,
    // ... map common options
}

// --- Ollama Response Types ---

#[derive(Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,  // ISO 8601
    pub message: OllamaMessage,
    pub done: bool,
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub prompt_eval_count: Option<u32>,
    pub eval_count: Option<u32>,
}

#[derive(Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
}
```

### 2. Request Translation

`fn ollama_to_openai(req: &OllamaChatRequest) -> serde_json::Value`:
- Map `messages` directly (same role/content structure)
- Map `options.temperature` → `temperature`
- Map `options.top_p` → `top_p`
- Map `options.num_predict` → `max_tokens`
- Set `stream: false`

### 3. Response Translation

`fn openai_to_ollama(resp: &serde_json::Value, model: &str) -> OllamaChatResponse`:
- Extract `choices[0].message.content` → `message.content`
- Extract `choices[0].message.role` → `message.role`
- Extract `choices[0].finish_reason` → `done_reason`
- Set `done: true`
- Map `usage.prompt_tokens` → `prompt_eval_count`
- Map `usage.completion_tokens` → `eval_count`
- Generate `created_at` as ISO 8601 timestamp

### 4. Handlers (`api/ollama.rs`)

**`chat`**: Translate Ollama request → OpenAI, proxy to llama-server, translate response back.

**`generate`**: Similar but uses `/v1/completions` instead of `/v1/chat/completions`. Ollama's generate uses a `prompt` field instead of `messages`.

**`tags`**: Call `model::list_models()` from the model manager, format as Ollama tags response. For now, just return the currently loaded model.

**`show`**: Return model info for the currently loaded model (name, size, path).

### 5. `/api/generate` Translation

Ollama's generate endpoint uses:
```json
{"model": "...", "prompt": "Hello", "stream": false}
```

This maps to OpenAI's chat completion with a single user message:
```json
{"messages": [{"role": "user", "content": "Hello"}], "stream": false}
```

## Tests

```rust
#[cfg(test)]
mod tests {
    // Translation tests
    fn test_ollama_to_openai_chat_translation() { ... }
    fn test_openai_to_ollama_response_translation() { ... }
    fn test_ollama_generate_to_openai_translation() { ... }
    fn test_ollama_options_mapping() { ... }

    // Integration tests with wiremock
    async fn test_api_chat_non_streaming() { ... }
    async fn test_api_generate_non_streaming() { ... }
    async fn test_api_tags_returns_loaded_model() { ... }
    async fn test_api_show_returns_model_info() { ... }
}
```

## Acceptance Criteria

- [ ] `POST /api/chat` with `stream: false` returns Ollama-format response
- [ ] `POST /api/generate` with `stream: false` works
- [ ] `GET /api/tags` returns model list in Ollama format
- [ ] `POST /api/show` returns model details
- [ ] Request translation correctly maps Ollama fields to OpenAI
- [ ] Response translation correctly maps OpenAI fields to Ollama
- [ ] Quality gate passes
