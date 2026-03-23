use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::post;
use serde_json::json;
use tokio::net::TcpListener;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Build a test proxy server pointing at the given mock llama-server.
async fn setup_proxy(mock_url: &str) -> SocketAddr {
    let state = llama_rs::api::AppState {
        config: Arc::new(llama_rs::config::Config::from_env()),
        llama_server_url: mock_url.to_string(),
        model_name: "test-model.gguf".to_string(),
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

/// Start a mini axum server that mimics llama-server's SSE streaming.
async fn start_sse_mock(chunks: Vec<&'static str>) -> SocketAddr {
    let router = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let chunks = chunks.clone();
            async move {
                let stream = async_stream::stream! {
                    for chunk in chunks {
                        yield Ok::<_, std::convert::Infallible>(
                            Event::default().data(chunk.to_string())
                        );
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                };
                Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(30)))
            }
        }),
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    addr
}

fn url(addr: SocketAddr, p: &str) -> String {
    format!("http://{addr}{p}")
}

// ─── Non-streaming tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_chat_completions_passthrough() {
    let mock_server = MockServer::start().await;

    let canned_response = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(&canned_response)
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
}

#[tokio::test]
async fn test_list_models_returns_model_info() {
    let mock_server = MockServer::start().await;
    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client.get(url(addr, "/v1/models")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "test-model.gguf");
    assert_eq!(body["data"][0]["owned_by"], "local");
}

#[tokio::test]
async fn test_chat_completions_preserves_error_status() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {"message": "bad request", "type": "invalid_request_error"}
        })))
        .mount(&mock_server)
        .await;

    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_cors_headers_present() {
    let mock_server = MockServer::start().await;
    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(url(addr, "/health"))
        .header("Origin", "http://example.com")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert!(resp.headers().contains_key("access-control-allow-origin"));
}

#[tokio::test]
async fn test_health_endpoint() {
    let mock_server = MockServer::start().await;
    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client.get(url(addr, "/health")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
}

// ─── Ollama API tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ollama_chat_non_streaming() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        })))
        .mount(&mock_server)
        .await;

    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["message"]["content"], "Hi there!");
    assert_eq!(body["message"]["role"], "assistant");
    assert_eq!(body["done"], true);
    assert_eq!(body["done_reason"], "stop");
}

#[tokio::test]
async fn test_ollama_generate_non_streaming() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Because of Rayleigh scattering."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        })))
        .mount(&mock_server)
        .await;

    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/generate"))
        .json(&json!({
            "model": "llama3",
            "prompt": "Why is the sky blue?",
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["response"], "Because of Rayleigh scattering.");
    assert_eq!(body["done"], true);
}

#[tokio::test]
async fn test_ollama_tags_returns_loaded_model() {
    let mock_server = MockServer::start().await;
    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client.get(url(addr, "/api/tags")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["models"][0]["name"], "test-model.gguf:latest");
}

#[tokio::test]
async fn test_ollama_show_returns_model_info() {
    let mock_server = MockServer::start().await;
    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/show"))
        .json(&json!({"model": "test-model.gguf"}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["modelfile"]
            .as_str()
            .unwrap()
            .contains("test-model.gguf")
    );
}

// ─── Ollama NDJSON Streaming tests ───────────────────────────────────────────

#[tokio::test]
async fn test_ollama_chat_streaming_ndjson() {
    let chunk1 = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
    let chunk2 = r#"{"choices":[{"delta":{"content":" world"},"finish_reason":null}]}"#;

    let mock_addr = start_sse_mock(vec![chunk1, chunk2]).await;
    let proxy_addr = setup_proxy(&format!("http://{mock_addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(proxy_addr, "/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("application/x-ndjson")
    );

    let body = resp.text().await.unwrap();
    // Parse each NDJSON line
    let lines: Vec<serde_json::Value> = body
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    // Should have content lines + done line
    assert!(
        lines.len() >= 2,
        "Expected at least 2 lines, got: {lines:?}"
    );

    // First content line
    assert_eq!(lines[0]["message"]["content"], "Hello");
    assert_eq!(lines[0]["done"], false);

    // Second content line
    assert_eq!(lines[1]["message"]["content"], " world");
    assert_eq!(lines[1]["done"], false);

    // Last line should be done
    let last = lines.last().unwrap();
    assert_eq!(last["done"], true);
}

#[tokio::test]
async fn test_ollama_generate_streaming_ndjson() {
    let chunk1 = r#"{"choices":[{"delta":{"content":"Because"},"finish_reason":null}]}"#;

    let mock_addr = start_sse_mock(vec![chunk1]).await;
    let proxy_addr = setup_proxy(&format!("http://{mock_addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(proxy_addr, "/api/generate"))
        .json(&json!({
            "model": "llama3",
            "prompt": "Why?",
            "stream": true
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

    assert!(!lines.is_empty());
    // Generate uses "response" field, not "message"
    assert_eq!(lines[0]["response"], "Because");
    assert_eq!(lines[0]["done"], false);

    let last = lines.last().unwrap();
    assert_eq!(last["done"], true);
}

#[tokio::test]
async fn test_ollama_chat_ndjson_content_type() {
    let mock_addr = start_sse_mock(vec![
        r#"{"choices":[{"delta":{"content":"x"},"finish_reason":null}]}"#,
    ])
    .await;
    let proxy_addr = setup_proxy(&format!("http://{mock_addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(proxy_addr, "/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        ct.contains("application/x-ndjson"),
        "Expected application/x-ndjson, got: {ct}"
    );
}

// ─── Frontend Compatibility tests ────────────────────────────────────────────

/// Test with a real OpenWebUI-style Ollama chat request payload.
#[tokio::test]
async fn test_openwebui_ollama_request() {
    let payload: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/openwebui_chat.json")).unwrap();

    // OpenWebUI sends stream: true to /api/chat by default
    let chunk1 = r#"{"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}"#;
    let mock_addr = start_sse_mock(vec![chunk1]).await;
    let proxy_addr = setup_proxy(&format!("http://{mock_addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(proxy_addr, "/api/chat"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(ct.contains("application/x-ndjson"));

    let body = resp.text().await.unwrap();
    let lines: Vec<serde_json::Value> = body
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert!(!lines.is_empty());
    // Verify content line
    let content_line = lines.iter().find(|l| !l["done"].as_bool().unwrap_or(true));
    assert!(content_line.is_some(), "Should have content lines");
    assert_eq!(content_line.unwrap()["message"]["content"], "Hi");
}

/// Test with a real LibreChat-style OpenAI chat request payload.
#[tokio::test]
async fn test_librechat_openai_request() {
    let payload: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/librechat_chat.json")).unwrap();

    // LibreChat sends stream: true to /v1/chat/completions
    let chunk1 = r#"{"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"content":"4"},"finish_reason":null}]}"#;
    let mock_addr = start_sse_mock(vec![chunk1]).await;
    let proxy_addr = setup_proxy(&format!("http://{mock_addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(proxy_addr, "/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(ct.contains("text/event-stream"));

    let body = resp.text().await.unwrap();
    assert!(body.contains("4"), "Should contain response content");
}

/// OpenWebUI also queries /api/tags to discover models.
#[tokio::test]
async fn test_openwebui_tags_discovery() {
    let mock_server = MockServer::start().await;
    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client.get(url(addr, "/api/tags")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["models"].is_array());
    assert!(!body["models"].as_array().unwrap().is_empty());
}

/// LibreChat queries /v1/models to discover models.
#[tokio::test]
async fn test_librechat_models_discovery() {
    let mock_server = MockServer::start().await;
    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client.get(url(addr, "/v1/models")).send().await.unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert!(!body["data"].as_array().unwrap().is_empty());
    assert_eq!(body["data"][0]["object"], "model");
}

// ─── Edge case tests ─────────────────────────────────────────────────────────

/// Non-streaming Ollama /api/chat with options (temperature, top_p).
#[tokio::test]
async fn test_ollama_chat_with_options() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        })))
        .mount(&mock_server)
        .await;

    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "options": {"temperature": 0.5, "top_p": 0.8, "num_predict": 100}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["message"]["content"], "ok");
    assert_eq!(body["done"], true);
}

/// Ollama /api/generate with system prompt.
#[tokio::test]
async fn test_ollama_generate_with_system() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"role": "assistant", "content": "result"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        })))
        .mount(&mock_server)
        .await;

    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/generate"))
        .json(&json!({
            "model": "llama3",
            "prompt": "Tell me a joke",
            "system": "You are a comedian.",
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["response"], "result");
}

/// Test that upstream 500 errors are properly relayed for Ollama endpoints.
#[tokio::test]
async fn test_ollama_upstream_error_relayed() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    let addr = setup_proxy(&mock_server.uri()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(addr, "/api/chat"))
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 500);
}

// ─── SSE Streaming tests ────────────────────────────────────────────────────

#[tokio::test]
async fn test_sse_stream_forwards_events() {
    let chunk1 =
        r#"{"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}"#;
    let chunk2 =
        r#"{"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}"#;

    let mock_addr = start_sse_mock(vec![chunk1, chunk2]).await;
    let proxy_addr = setup_proxy(&format!("http://{mock_addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(proxy_addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream")
    );

    // Read the full body and check SSE events are present
    let body = resp.text().await.unwrap();
    assert!(body.contains("Hello"), "body should contain Hello: {body}");
    assert!(body.contains(" world"), "body should contain world: {body}");
}

#[tokio::test]
async fn test_sse_stream_content_type() {
    let mock_addr = start_sse_mock(vec![r#"{"test":true}"#]).await;
    let proxy_addr = setup_proxy(&format!("http://{mock_addr}")).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(url(proxy_addr, "/v1/chat/completions"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        ct.contains("text/event-stream"),
        "Expected text/event-stream, got: {ct}"
    );
}
