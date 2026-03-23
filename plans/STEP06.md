# Step 06: OpenAI Streaming (SSE Passthrough)

## Objective
Implement streaming support for `/v1/chat/completions` when `stream: true`. Forward SSE events from llama-server to the client with zero translation.

## Instructions

### 1. Streaming Detection

In the `chat_completions` handler, branch on the `stream` field:
```rust
if request.stream {
    stream_chat_completions(state, body).await
} else {
    proxy_chat_completions(state, body).await
}
```

### 2. SSE Stream Proxy (`api/stream/sse.rs`)

```rust
pub async fn stream_from_upstream(
    client: &reqwest::Client,
    url: &str,
    body: serde_json::Value,
) -> Result<impl Stream<Item = Result<Event, anyhow::Error>>> {
    let response = client
        .post(url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;

    let stream = async_stream::stream! {
        let mut bytes_stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = bytes_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Parse SSE lines from buffer
            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in event_text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            return; // End of stream
                        }
                        yield Ok(Event::default().data(data.to_string()));
                    }
                }
            }
        }
    };

    Ok(stream)
}
```

### 3. Axum SSE Response

```rust
async fn stream_chat_completions(
    state: AppState,
    body: serde_json::Value,
) -> Result<Sse<impl Stream<Item = Result<Event, anyhow::Error>>>, StatusCode> {
    let stream = stream_from_upstream(
        &state.http_client,
        &format!("{}/v1/chat/completions", state.llama_server_url),
        body,
    ).await
    .map_err(|_| StatusCode::BAD_GATEWAY)?;

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
    ))
}
```

### 4. Key Details

- The `Content-Type: text/event-stream` header is set automatically by axum's `Sse` response type
- Buffer SSE parsing must handle partial chunks (TCP doesn't guarantee message boundaries)
- The `[DONE]` sentinel terminates the stream — don't forward it as an SSE event, just close
- Keep-alive comments prevent proxy/LB timeout on slow generation

## Tests

```rust
#[cfg(test)]
mod tests {
    // Mock llama-server streaming response with wiremock
    async fn test_sse_stream_forwards_events() {
        // Set up wiremock to return SSE response
        // Connect to our proxy with streaming
        // Assert we receive the same events
    }

    async fn test_sse_stream_ends_on_done() { ... }

    async fn test_sse_handles_partial_chunks() {
        // Simulate SSE data split across TCP chunks
    }

    // Unit test for SSE parsing
    fn test_parse_sse_line() { ... }
    fn test_parse_sse_done_sentinel() { ... }
}
```

## Acceptance Criteria

- [ ] `POST /v1/chat/completions` with `stream: true` returns `text/event-stream`
- [ ] SSE events from llama-server are forwarded in real-time
- [ ] Stream terminates cleanly on `[DONE]`
- [ ] Partial TCP chunks are handled correctly (buffer-based parsing)
- [ ] Keep-alive comments are sent during slow generation
- [ ] Works with `curl` and standard SSE clients
- [ ] Quality gate passes
