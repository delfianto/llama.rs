use axum::body::Bytes;
use futures::StreamExt;
use tracing::debug;

/// Convert an upstream SSE byte stream (from llama-server) into an NDJSON stream
/// in Ollama format.
///
/// For `/api/chat`: each NDJSON line has `{"model":..,"message":{"role":"assistant","content":"token"},"done":false}`.
/// For `/api/generate`: each NDJSON line has `{"model":..,"response":"token","done":false}`.
///
/// The final line has `"done": true` with optional `"done_reason"`.
pub fn sse_to_ndjson_chat_stream(
    upstream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>>,
    model_name: String,
) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>> {
    let model = model_name;

    async_stream::stream! {
        let mut buffer = String::new();
        let mut stream = std::pin::pin!(upstream);
        let mut emitted_done = false;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("Upstream stream error: {e}");
                    break;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in event_text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            debug!("NDJSON chat: received [DONE]");
                            if !emitted_done {
                                let final_line = build_chat_done_line(&model, None);
                                yield Ok(Bytes::from(final_line));
                            }
                            return;
                        }

                        if let Some((ndjson_line, is_done)) = openai_chunk_to_ndjson_chat(data, &model) {
                            yield Ok(Bytes::from(ndjson_line));
                            if is_done {
                                emitted_done = true;
                            }
                        }
                    }
                }
            }
        }

        // If stream ended without [DONE], emit final line anyway
        if !emitted_done {
            let final_line = build_chat_done_line(&model, None);
            yield Ok(Bytes::from(final_line));
        }
    }
}

/// Same as chat but for `/api/generate` format (uses `response` field instead of `message`).
pub fn sse_to_ndjson_generate_stream(
    upstream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>>,
    model_name: String,
) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>> {
    let model = model_name;

    async_stream::stream! {
        let mut buffer = String::new();
        let mut stream = std::pin::pin!(upstream);
        let mut emitted_done = false;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("Upstream stream error: {e}");
                    break;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in event_text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            debug!("NDJSON generate: received [DONE]");
                            if !emitted_done {
                                let final_line = build_generate_done_line(&model, None);
                                yield Ok(Bytes::from(final_line));
                            }
                            return;
                        }

                        if let Some((ndjson_line, is_done)) = openai_chunk_to_ndjson_generate(data, &model) {
                            yield Ok(Bytes::from(ndjson_line));
                            if is_done {
                                emitted_done = true;
                            }
                        }
                    }
                }
            }
        }

        if !emitted_done {
            let final_line = build_generate_done_line(&model, None);
            yield Ok(Bytes::from(final_line));
        }
    }
}

/// Convert a single OpenAI `chat.completion.chunk` JSON string into an Ollama
/// chat NDJSON line. Returns `(line, is_done)` or `None` if the chunk has no content.
///
/// Handles both standard `delta.content` and reasoning model `delta.reasoning_content`.
fn openai_chunk_to_ndjson_chat(data: &str, model: &str) -> Option<(String, bool)> {
    let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
    let choice = &parsed["choices"][0];
    let delta = &choice["delta"];

    // Check for finish_reason — emit done line
    if let Some(reason) = choice["finish_reason"].as_str() {
        return Some((build_chat_done_line(model, Some(reason)), true));
    }

    // Extract content from delta — check both content and reasoning_content
    let content = delta["content"]
        .as_str()
        .or_else(|| delta["reasoning_content"].as_str())
        .unwrap_or("");

    // First chunk with just role and no content, skip
    if content.is_empty() {
        return None;
    }

    let now = chrono::Utc::now().to_rfc3339();
    let line = serde_json::json!({
        "model": model,
        "created_at": now,
        "message": {"role": "assistant", "content": content},
        "done": false,
    });
    Some((format!("{line}\n"), false))
}

/// Convert a single OpenAI chunk into an Ollama generate NDJSON line.
fn openai_chunk_to_ndjson_generate(data: &str, model: &str) -> Option<(String, bool)> {
    let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
    let choice = &parsed["choices"][0];
    let delta = &choice["delta"];

    if let Some(reason) = choice["finish_reason"].as_str() {
        return Some((build_generate_done_line(model, Some(reason)), true));
    }

    let content = delta["content"]
        .as_str()
        .or_else(|| delta["reasoning_content"].as_str())
        .unwrap_or("");

    if content.is_empty() {
        return None;
    }

    let now = chrono::Utc::now().to_rfc3339();
    let line = serde_json::json!({
        "model": model,
        "created_at": now,
        "response": content,
        "done": false,
    });
    Some((format!("{line}\n"), false))
}

fn build_chat_done_line(model: &str, reason: Option<&str>) -> String {
    let now = chrono::Utc::now().to_rfc3339();
    let mut line = serde_json::json!({
        "model": model,
        "created_at": now,
        "message": {"role": "assistant", "content": ""},
        "done": true,
    });
    if let Some(r) = reason {
        line["done_reason"] = serde_json::json!(r);
    }
    format!("{line}\n")
}

fn build_generate_done_line(model: &str, reason: Option<&str>) -> String {
    let now = chrono::Utc::now().to_rfc3339();
    let mut line = serde_json::json!({
        "model": model,
        "created_at": now,
        "response": "",
        "done": true,
    });
    if let Some(r) = reason {
        line["done_reason"] = serde_json::json!(r);
    }
    format!("{line}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_chunk_to_ollama_ndjson_chat() {
        let chunk = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let (line, is_done) = openai_chunk_to_ndjson_chat(chunk, "llama3").unwrap();
        assert!(!is_done);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["message"]["content"], "Hello");
        assert_eq!(parsed["message"]["role"], "assistant");
        assert_eq!(parsed["done"], false);
        assert_eq!(parsed["model"], "llama3");
    }

    #[test]
    fn test_openai_chunk_to_ollama_ndjson_generate() {
        let chunk = r#"{"choices":[{"delta":{"content":"World"},"finish_reason":null}]}"#;
        let (line, is_done) = openai_chunk_to_ndjson_generate(chunk, "llama3").unwrap();
        assert!(!is_done);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["response"], "World");
        assert_eq!(parsed["done"], false);
    }

    #[test]
    fn test_final_chunk_sets_done_true_chat() {
        let chunk = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let (line, is_done) = openai_chunk_to_ndjson_chat(chunk, "llama3").unwrap();
        assert!(is_done);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["done"], true);
        assert_eq!(parsed["done_reason"], "stop");
    }

    #[test]
    fn test_final_chunk_sets_done_true_generate() {
        let chunk = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let (line, is_done) = openai_chunk_to_ndjson_generate(chunk, "llama3").unwrap();
        assert!(is_done);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["done"], true);
        assert_eq!(parsed["done_reason"], "stop");
    }

    #[test]
    fn test_role_only_chunk_skipped() {
        let chunk = r#"{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        assert!(openai_chunk_to_ndjson_chat(chunk, "llama3").is_none());
    }

    #[test]
    fn test_done_line_format() {
        let line = build_chat_done_line("llama3", Some("stop"));
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["done"], true);
        assert_eq!(parsed["done_reason"], "stop");
        assert_eq!(parsed["message"]["content"], "");
    }

    #[test]
    fn test_null_content_chunk_skipped() {
        // llama-server sends content: null in the first chunk for some models
        let chunk =
            r#"{"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}"#;
        assert!(openai_chunk_to_ndjson_chat(chunk, "llama3").is_none());
    }

    #[test]
    fn test_reasoning_content_chunk() {
        let chunk =
            r#"{"choices":[{"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}"#;
        let (line, is_done) = openai_chunk_to_ndjson_chat(chunk, "llama3").unwrap();
        assert!(!is_done);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["message"]["content"], "thinking");
    }
}
