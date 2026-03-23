# Step 08: Ollama Streaming (NDJSON)

## Objective
Implement streaming for Ollama endpoints. Ollama uses NDJSON (newline-delimited JSON), not SSE. Each token is a JSON object on its own line.

## Instructions

### 1. Key Difference from OpenAI SSE

Ollama streaming format:
```
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":"Hello"},"done":false}\n
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":" world"},"done":false}\n
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":...}\n
```

- Content-Type: `application/x-ndjson`
- No `data:` prefix
- No `[DONE]` sentinel — completion signaled by `"done": true`
- Each line is a complete, parseable JSON object

### 2. NDJSON Stream Formatter (`api/stream/ndjson.rs`)

```rust
pub fn sse_to_ndjson_stream(
    sse_stream: impl Stream<Item = Result<Bytes, reqwest::Error>>,
    model_name: &str,
) -> impl Stream<Item = Result<Bytes, anyhow::Error>> {
    let model = model_name.to_string();

    async_stream::stream! {
        let mut buffer = String::new();

        // Process upstream SSE events
        // For each "data: {json}" line from llama-server:
        //   1. Parse the OpenAI chunk JSON
        //   2. Extract delta.content
        //   3. Build Ollama NDJSON line
        //   4. Yield as Bytes

        // When we see "data: [DONE]", emit final line with done: true
    }
}
```

### 3. Translation Per Chunk

For each SSE `chat.completion.chunk` from llama-server:
```json
{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}
```

Emit Ollama NDJSON line:
```json
{"model":"llama3","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false}
```

On the final chunk (where `finish_reason` is `"stop"`), or on `[DONE]`:
```json
{"model":"llama3","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}
```

### 4. Axum Response

Use `axum::body::Body::from_stream()` for NDJSON — NOT `Sse`, since Ollama doesn't use SSE:

```rust
async fn stream_chat(state: AppState, req: OllamaChatRequest) -> Response {
    let openai_body = ollama_to_openai(&req);
    openai_body["stream"] = json!(true);

    let upstream = state.http_client
        .post(format!("{}/v1/chat/completions", state.llama_server_url))
        .json(&openai_body)
        .send()
        .await?;

    let ndjson_stream = sse_to_ndjson_stream(
        upstream.bytes_stream(),
        &req.model,
    );

    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .header("Transfer-Encoding", "chunked")
        .body(Body::from_stream(ndjson_stream))
        .unwrap()
}
```

### 5. Update Ollama Handlers

In `api/ollama.rs`, branch on `stream` field (default true for Ollama!):
- `stream: true` (default) → NDJSON streaming
- `stream: false` → non-streaming response from Step 07

**Note**: Ollama defaults `stream` to `true`, opposite of OpenAI which defaults to `false`.

## Tests

```rust
#[cfg(test)]
mod tests {
    // Unit: SSE chunk to NDJSON line
    fn test_openai_chunk_to_ollama_ndjson() { ... }
    fn test_final_chunk_sets_done_true() { ... }
    fn test_done_sentinel_produces_final_line() { ... }

    // Integration with wiremock SSE mock
    async fn test_ollama_chat_streaming() {
        // Mock llama-server returning SSE
        // POST /api/chat with stream: true
        // Read NDJSON lines
        // Verify format and content
    }

    async fn test_ollama_generate_streaming() { ... }

    // Edge cases
    async fn test_ollama_streaming_default_true() {
        // POST /api/chat without stream field
        // Should stream by default
    }
}
```

## Acceptance Criteria

- [ ] `POST /api/chat` streams NDJSON by default
- [ ] Each NDJSON line is valid JSON matching Ollama format
- [ ] Stream terminates with `done: true` line
- [ ] Content-Type is `application/x-ndjson`
- [ ] `POST /api/generate` streaming works similarly
- [ ] Compatible with OpenWebUI's Ollama connection
- [ ] Quality gate passes
