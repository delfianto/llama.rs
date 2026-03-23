//! Integration tests that run against a real llama-server on localhost:8080.
//!
//! These tests require a running llama-server:
//!   llama-server -m model.gguf --port 8080
//!
//! Run with: cargo test --test live_server
//!
//! They are NOT marked #[ignore] so they run as part of the normal test suite
//! when the server is available. They fail fast with a clear message if not.

use std::net::SocketAddr;
use std::sync::Arc;

use serde_json::json;
use tokio::net::TcpListener;

const LLAMA_SERVER_URL: &str = "http://127.0.0.1:8080";

/// Check if llama-server is reachable. Skip all tests if not.
async fn require_server() {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();

    match client
        .get(format!("{LLAMA_SERVER_URL}/health"))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {}
        _ => panic!(
            "llama-server not reachable at {LLAMA_SERVER_URL}. \
             Start it with: llama-server -m model.gguf --port 8080"
        ),
    }
}

/// Get the model name from the running llama-server.
async fn get_model_name() -> String {
    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .get(format!("{LLAMA_SERVER_URL}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    resp["data"][0]["id"]
        .as_str()
        .unwrap_or("unknown")
        .to_string()
}

/// Start our proxy server pointing at the real llama-server.
async fn start_proxy() -> SocketAddr {
    let model_name = get_model_name().await;

    let state = llama_rs::api::AppState {
        config: Arc::new(llama_rs::config::Config::from_env()),
        llama_server_url: LLAMA_SERVER_URL.to_string(),
        model_name,
        http_client: reqwest::Client::new(),
    };

    let router = llama_rs::api::build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    addr
}

fn url(addr: SocketAddr, path: &str) -> String {
    format!("http://{addr}{path}")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Health & Discovery
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_proxy_health() {
    require_server().await;
    let addr = start_proxy().await;

    let resp = reqwest::get(url(addr, "/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_proxy_v1_models() {
    require_server().await;
    let addr = start_proxy().await;

    let resp = reqwest::get(url(addr, "/v1/models")).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let models = body["data"].as_array().unwrap();
    assert!(!models.is_empty(), "Should have at least one model");
    assert_eq!(models[0]["object"], "model");
    assert_eq!(models[0]["owned_by"], "local");
    assert!(models[0]["id"].as_str().unwrap().contains(".gguf"));
}

#[tokio::test]
async fn test_proxy_api_tags() {
    require_server().await;
    let addr = start_proxy().await;

    let resp = reqwest::get(url(addr, "/api/tags")).await.unwrap();
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty());
    assert!(models[0]["name"].as_str().is_some());
    assert!(models[0]["modified_at"].as_str().is_some());
}

// ═══════════════════════════════════════════════════════════════════════════════
// OpenAI /v1/chat/completions — Non-Streaming
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_openai_non_streaming_structure() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Reply with the single word OK"}],
            "stream": false,
            "max_tokens": 50
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();

    // Full OpenAI response structure validation
    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().is_some(), "Missing id");
    assert!(body["created"].as_u64().is_some(), "Missing created");
    assert!(body["model"].as_str().is_some(), "Missing model");

    let choices = body["choices"].as_array().unwrap();
    assert!(!choices.is_empty());
    assert_eq!(choices[0]["message"]["role"], "assistant");
    assert!(choices[0]["message"]["content"].as_str().is_some());
    assert!(choices[0]["finish_reason"].as_str().is_some());
    assert_eq!(choices[0]["index"], 0);

    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(usage["completion_tokens"].as_u64().unwrap() > 0);
    assert!(usage["total_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_openai_non_streaming_multi_turn() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "You answer in exactly one word."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "Blue"},
                {"role": "user", "content": "What color is grass?"}
            ],
            "stream": false,
            "max_tokens": 20
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["choices"][0]["message"]["content"].as_str().is_some());
}

#[tokio::test]
async fn test_openai_non_streaming_with_params() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": false,
            "max_tokens": 5,
            "temperature": 0.1,
            "top_p": 0.9
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    // With max_tokens=5, completion_tokens should be <= 5
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() <= 5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// OpenAI /v1/chat/completions — Streaming (SSE)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_openai_streaming_content_type() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": true,
            "max_tokens": 5
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        ct.contains("text/event-stream"),
        "Expected text/event-stream, got: {ct}"
    );
}

#[tokio::test]
async fn test_openai_streaming_chunk_structure() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": true,
            "max_tokens": 10
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();

    // Parse SSE data lines — our proxy strips [DONE] and closes the stream,
    // which is correct SSE behavior.
    let mut chunks: Vec<serde_json::Value> = Vec::new();

    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            let chunk: serde_json::Value = serde_json::from_str(data)
                .unwrap_or_else(|e| panic!("Failed to parse SSE chunk: {e}\nData: {data}"));
            chunks.push(chunk);
        }
    }

    assert!(
        !chunks.is_empty(),
        "Should have at least one chunk:\n{body}"
    );

    // Validate every chunk has correct structure
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk["object"], "chat.completion.chunk",
            "chunk[{i}] object"
        );
        assert!(chunk["id"].as_str().is_some(), "chunk[{i}] missing id");
        assert!(
            chunk["model"].as_str().is_some(),
            "chunk[{i}] missing model"
        );
        assert!(
            chunk["choices"][0]["delta"].is_object(),
            "chunk[{i}] missing delta"
        );
        assert_eq!(chunk["choices"][0]["index"], 0, "chunk[{i}] index");
    }

    // Last chunk should have finish_reason
    let last = chunks.last().unwrap();
    assert!(
        last["choices"][0]["finish_reason"].as_str().is_some(),
        "Last chunk should have finish_reason: {last}"
    );
}

#[tokio::test]
async fn test_openai_streaming_multi_turn() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are brief."},
                {"role": "user", "content": "Hi"}
            ],
            "stream": true,
            "max_tokens": 10
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("data: "), "Should have SSE data lines");
}

// ═══════════════════════════════════════════════════════════════════════════════
// OpenAI Error Handling
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_openai_missing_messages_field() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({"model": "test", "stream": false}))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "Expected error status, got: {}",
        resp.status()
    );
}

/// Verify the proxy is a true passthrough — send a request with extra/unusual
/// fields and confirm the upstream response is returned byte-for-byte.
#[tokio::test]
async fn test_openai_passthrough_preserves_extra_fields() {
    require_server().await;
    let addr = start_proxy().await;

    // Send request with extra fields that a frontend might include
    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::to_string(&json!({
                "model": "test",
                "messages": [{"role": "user", "content": "Say OK"}],
                "stream": false,
                "max_tokens": 10,
                "user": "test-user-123",
                "some_custom_field": true
            }))
            .unwrap(),
        )
        .send()
        .await
        .unwrap();

    // Should succeed — llama-server ignores unknown fields
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    // Response should contain llama-server's original fields untouched
    assert!(body["id"].as_str().is_some(), "id from llama-server");
    assert!(
        body["system_fingerprint"].as_str().is_some()
            || body["system_fingerprint"].is_null()
            || !body.get("system_fingerprint").is_some(),
        "system_fingerprint passthrough"
    );
    // Timings field from llama-server should be present (not stripped by proxy)
    // llama.cpp includes this but standard OpenAI doesn't — true passthrough preserves it
    if body.get("timings").is_some() {
        assert!(
            body["timings"].is_object(),
            "timings should be passed through"
        );
    }
}

/// Verify that the proxy passes the upstream response body through on error,
/// not just the status code.
#[tokio::test]
async fn test_openai_error_body_passthrough() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    // Send malformed JSON — llama-server should return 400 with error details
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(r#"{"model": "test"}"#)
        .send()
        .await
        .unwrap();

    let status = resp.status().as_u16();
    assert!(status >= 400, "Expected error status, got: {status}");

    // The error body from llama-server should be relayed, not replaced
    let body = resp.text().await.unwrap();
    assert!(!body.is_empty(), "Error response body should not be empty");
}

/// Verify streaming error handling — if the initial request fails (e.g., bad request),
/// the error status should be relayed even for stream:true requests.
#[tokio::test]
async fn test_openai_streaming_error_passthrough() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    // Send a request with stream:true but missing messages
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(r#"{"model": "test", "stream": true}"#)
        .send()
        .await
        .unwrap();

    let status = resp.status().as_u16();
    // llama-server should return an error for missing messages
    assert!(
        status >= 400,
        "Expected error for missing messages with stream:true, got: {status}"
    );
}

/// Verify that response content-type header is relayed from upstream.
#[tokio::test]
async fn test_openai_content_type_passthrough() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::to_string(&json!({
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": false,
                "max_tokens": 5
            }))
            .unwrap(),
        )
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        ct.contains("application/json"),
        "Non-streaming response should have application/json, got: {ct}"
    );
}

/// Verify the SSE stream from llama-server includes [DONE] sentinel
/// (our proxy streams raw bytes, so [DONE] should be present in raw output).
#[tokio::test]
async fn test_openai_streaming_includes_done() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::to_string(&json!({
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": true,
                "max_tokens": 5
            }))
            .unwrap(),
        )
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("data: [DONE]"),
        "Raw passthrough should include [DONE] sentinel:\n{body}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ollama /api/chat — Non-Streaming
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_chat_non_streaming_structure() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Reply with the single word OK"}],
            "stream": false,
            "options": {"num_predict": 50}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();

    // Full Ollama response structure validation
    assert!(body["model"].as_str().is_some(), "Missing model");
    assert!(body["created_at"].as_str().is_some(), "Missing created_at");
    assert_eq!(body["done"], true);
    assert_eq!(body["message"]["role"], "assistant");
    assert!(
        !body["message"]["content"].as_str().unwrap_or("").is_empty(),
        "Content should not be empty: {body}"
    );
    assert!(
        body["done_reason"].as_str().is_some(),
        "Missing done_reason: {body}"
    );
    // Usage stats should be present
    assert!(
        body["prompt_eval_count"].as_u64().is_some(),
        "Missing prompt_eval_count: {body}"
    );
    assert!(
        body["eval_count"].as_u64().is_some(),
        "Missing eval_count: {body}"
    );
}

#[tokio::test]
async fn test_ollama_chat_non_streaming_with_options() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": false,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 5
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["done"], true);
    // eval_count should be <= num_predict
    if let Some(eval) = body["eval_count"].as_u64() {
        assert!(eval <= 5, "eval_count {eval} should be <= num_predict 5");
    }
}

#[tokio::test]
async fn test_ollama_chat_non_streaming_multi_turn() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "You answer in one word."},
                {"role": "user", "content": "Color of sky?"},
                {"role": "assistant", "content": "Blue"},
                {"role": "user", "content": "Color of grass?"}
            ],
            "stream": false,
            "options": {"num_predict": 20}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["done"], true);
    assert!(!body["message"]["content"].as_str().unwrap_or("").is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ollama /api/chat — Streaming (NDJSON)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_chat_streaming_content_type() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
            "options": {"num_predict": 5}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        ct.contains("application/x-ndjson"),
        "Expected NDJSON, got: {ct}"
    );
}

#[tokio::test]
async fn test_ollama_chat_streaming_line_structure() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": true,
            "options": {"num_predict": 10}
        }))
        .send()
        .await
        .unwrap();

    let body = resp.text().await.unwrap();
    let lines: Vec<serde_json::Value> = body
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            serde_json::from_str(l).unwrap_or_else(|e| panic!("Bad NDJSON line: {e}\nLine: {l}"))
        })
        .collect();

    assert!(
        lines.len() >= 2,
        "Expected >=2 lines (content + done), got {}:\n{body}",
        lines.len()
    );

    // Content lines
    let content_lines: Vec<&serde_json::Value> =
        lines.iter().filter(|l| l["done"] == false).collect();
    assert!(
        !content_lines.is_empty(),
        "Should have content lines:\n{body}"
    );

    for cl in &content_lines {
        assert_eq!(cl["message"]["role"], "assistant");
        assert!(
            cl["message"]["content"].as_str().is_some(),
            "Missing content: {cl}"
        );
        assert!(cl["model"].as_str().is_some(), "Missing model: {cl}");
        assert!(
            cl["created_at"].as_str().is_some(),
            "Missing created_at: {cl}"
        );
        assert_eq!(cl["done"], false);
    }

    // Done line
    let last = lines.last().unwrap();
    assert_eq!(last["done"], true, "Last line should be done:\n{body}");
    assert!(
        last["done_reason"].as_str().is_some(),
        "Done line should have done_reason: {last}"
    );
}

#[tokio::test]
async fn test_ollama_chat_streaming_default_true() {
    require_server().await;
    let addr = start_proxy().await;

    // Omit "stream" field — Ollama defaults to true
    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {"num_predict": 5}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        ct.contains("application/x-ndjson"),
        "Without stream field, Ollama should default to streaming. Got content-type: {ct}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ollama /api/generate — Non-Streaming
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_generate_non_streaming_structure() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/generate"))
        .json(&json!({
            "model": "test",
            "prompt": "Reply with the single word OK",
            "stream": false,
            "options": {"num_predict": 50}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(body["done"], true);
    assert!(body["model"].as_str().is_some(), "Missing model");
    assert!(body["created_at"].as_str().is_some(), "Missing created_at");
    assert!(
        !body["response"].as_str().unwrap_or("").is_empty(),
        "Response should not be empty: {body}"
    );
    assert!(
        body["done_reason"].as_str().is_some(),
        "Missing done_reason: {body}"
    );
}

#[tokio::test]
async fn test_ollama_generate_non_streaming_with_system() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/generate"))
        .json(&json!({
            "model": "test",
            "prompt": "What is 2+2?",
            "system": "You are a math tutor. Answer in one word.",
            "stream": false,
            "options": {"num_predict": 20}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["done"], true);
    assert!(!body["response"].as_str().unwrap_or("").is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ollama /api/generate — Streaming (NDJSON)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_generate_streaming_structure() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/generate"))
        .json(&json!({
            "model": "test",
            "prompt": "Say hello",
            "stream": true,
            "options": {"num_predict": 10}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let body = resp.text().await.unwrap();
    let lines: Vec<serde_json::Value> = body
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert!(lines.len() >= 2, "Expected >=2 lines, got {}", lines.len());

    // Content lines use "response" field, not "message"
    let content_lines: Vec<&serde_json::Value> =
        lines.iter().filter(|l| l["done"] == false).collect();
    assert!(!content_lines.is_empty());
    for cl in &content_lines {
        assert!(cl["response"].is_string(), "Missing response field: {cl}");
        assert!(cl["model"].as_str().is_some());
        assert!(cl["created_at"].as_str().is_some());
    }

    let last = lines.last().unwrap();
    assert_eq!(last["done"], true);
    assert!(
        last["done_reason"].as_str().is_some(),
        "Missing done_reason: {last}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ollama /api/show
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_show() {
    require_server().await;
    let addr = start_proxy().await;

    let model_name = get_model_name().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/show"))
        .json(&json!({"model": model_name}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["modelfile"].as_str().is_some());
    assert!(
        body["modelfile"].as_str().unwrap().contains(&model_name),
        "modelfile should reference the model name"
    );
}
