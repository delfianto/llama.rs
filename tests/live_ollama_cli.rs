//! Integration tests that use the real `ollama` CLI binary to validate
//! our Ollama API emulation layer.
//!
//! Requires:
//!   1. llama-server running on localhost:8080
//!   2. ollama binary at /opt/homebrew/bin/ollama
//!
//! Run with: cargo test --test live_ollama_cli

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;

const LLAMA_SERVER_URL: &str = "http://127.0.0.1:8080";
const OLLAMA_BIN: &str = "/opt/homebrew/bin/ollama";

async fn require_deps() {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    match client
        .get(format!("{LLAMA_SERVER_URL}/health"))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {}
        _ => panic!("llama-server not reachable at {LLAMA_SERVER_URL}"),
    }

    if !std::path::Path::new(OLLAMA_BIN).exists() {
        panic!("ollama binary not found at {OLLAMA_BIN}");
    }
}

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

    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

/// Run ollama with timeout — the CLI can hang if something's unexpected.
async fn ollama_with_timeout(
    addr: SocketAddr,
    args: &[&str],
    timeout_secs: u64,
) -> Option<std::process::Output> {
    let addr_str = format!("http://{addr}");
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();

    let result = tokio::time::timeout(Duration::from_secs(timeout_secs), async move {
        tokio::task::spawn_blocking(move || {
            std::process::Command::new(OLLAMA_BIN)
                .args(&args)
                .env("OLLAMA_HOST", &addr_str)
                // Prevent ollama from trying to start its own server
                .env("OLLAMA_NOPRUNE", "1")
                .output()
                .ok()
        })
        .await
        .ok()
        .flatten()
    })
    .await
    .ok()
    .flatten();

    result
}

fn stdout_str(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr_str(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

// ═══════════════════════════════════════════════════════════════════════════════
// ollama list (via CLI)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_cli_list() {
    require_deps().await;
    let addr = start_proxy().await;

    let output = ollama_with_timeout(addr, &["list"], 10).await;

    match output {
        Some(o) if o.status.success() => {
            let out = stdout_str(&o);
            assert!(
                out.contains("NAME") || out.contains(".gguf"),
                "ollama list output should contain model info.\nstdout: {out}\nstderr: {}",
                stderr_str(&o)
            );
        }
        Some(o) => {
            let err = stderr_str(&o);
            let out = stdout_str(&o);
            // Some ollama versions may error on certain fields — log but don't fail hard
            // if the connection at least worked
            panic!(
                "ollama list failed (status {}).\nstdout: {out}\nstderr: {err}",
                o.status
            );
        }
        None => {
            panic!("ollama list timed out after 10s — likely hanging on connection");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ollama show (via CLI)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_cli_show() {
    require_deps().await;
    let addr = start_proxy().await;
    let model_name = get_model_name().await;

    let output = ollama_with_timeout(addr, &["show", &model_name], 10).await;

    match output {
        Some(o) => {
            let out = stdout_str(&o);
            let err = stderr_str(&o);
            // ollama show should at least get a response
            assert!(
                o.status.success() || out.contains("Model") || out.contains("FROM"),
                "ollama show should get a response.\nstdout: {out}\nstderr: {err}"
            );
        }
        None => {
            panic!("ollama show timed out after 10s");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ollama run (via CLI, non-interactive)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_ollama_cli_run() {
    require_deps().await;
    let addr = start_proxy().await;
    let model_name = get_model_name().await;

    // ollama run with a prompt as argument (non-interactive)
    let output = ollama_with_timeout(addr, &["run", &model_name, "Say hi briefly"], 30).await;

    match output {
        Some(o) => {
            let out = stdout_str(&o);
            let err = stderr_str(&o);
            assert!(
                o.status.success(),
                "ollama run should succeed.\nstdout: {out}\nstderr: {err}"
            );
            assert!(
                !out.trim().is_empty(),
                "ollama run should produce output.\nstdout: {out}\nstderr: {err}"
            );
        }
        None => {
            panic!("ollama run timed out after 30s");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Direct API validation (what ollama CLI does under the hood)
// ═══════════════════════════════════════════════════════════════════════════════

/// HEAD / — ollama CLI connectivity check
#[tokio::test]
async fn test_ollama_head_root() {
    require_deps().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client.head(format!("http://{addr}/")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
}

/// GET /api/version — ollama CLI version check
#[tokio::test]
async fn test_ollama_api_version() {
    require_deps().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/api/version"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["version"].as_str().is_some());
}

/// GET /api/tags — format validation
#[tokio::test]
async fn test_ollama_api_tags_format() {
    require_deps().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/api/tags"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();

    assert!(body["models"].is_array(), "Missing 'models' array: {body}");
    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty());

    for m in models {
        assert!(m["name"].as_str().is_some(), "Missing 'name': {m}");
        // modified_at must be a valid timestamp (not empty)
        let modified = m["modified_at"].as_str().unwrap_or("");
        assert!(!modified.is_empty(), "modified_at should not be empty: {m}");
        // Verify it parses as RFC 3339
        assert!(
            chrono::DateTime::parse_from_rfc3339(modified).is_ok(),
            "modified_at should be valid RFC 3339: {modified}"
        );
    }
}

/// POST /api/show — with name field (as ollama CLI sends it)
#[tokio::test]
async fn test_ollama_api_show_with_name_field() {
    require_deps().await;
    let addr = start_proxy().await;
    let model_name = get_model_name().await;

    let client = reqwest::Client::new();
    // This is what ollama CLI actually sends:
    let resp = client
        .post(format!("http://{addr}/api/show"))
        .json(&serde_json::json!({
            "model": "",
            "system": "",
            "template": "",
            "verbose": false,
            "options": null,
            "name": model_name
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["modelfile"].as_str().is_some());
}

/// POST /api/chat streaming — format validation
#[tokio::test]
async fn test_ollama_api_chat_streaming_format() {
    require_deps().await;
    let addr = start_proxy().await;
    let model_name = get_model_name().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/api/chat"))
        .json(&serde_json::json!({
            "model": model_name,
            "messages": [{"role": "user", "content": "Say hi"}],
            "options": {"num_predict": 5}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let body = resp.text().await.unwrap();
    let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty(), "Should have NDJSON lines");

    for (i, line) in lines.iter().enumerate() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Line {i} not valid JSON: {e}\nLine: {line}"));
        assert!(
            parsed["model"].as_str().is_some(),
            "Line {i} missing 'model'"
        );
        assert!(parsed["done"].is_boolean(), "Line {i} missing 'done'");

        if parsed["done"] == false {
            assert!(
                parsed["message"].is_object(),
                "Content line {i} missing 'message'"
            );
            assert!(parsed["message"]["role"].as_str().is_some());
        }
    }

    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(last["done"], true);
}

/// POST /api/generate — what ollama run actually sends
#[tokio::test]
async fn test_ollama_api_generate_as_cli_sends() {
    require_deps().await;
    let addr = start_proxy().await;
    let model_name = get_model_name().await;

    let client = reqwest::Client::new();
    // Exact payload ollama CLI sends for `ollama run model "prompt"`:
    let resp = client
        .post(format!("http://{addr}/api/generate"))
        .json(&serde_json::json!({
            "model": model_name,
            "prompt": "Say hello\n",
            "suffix": "",
            "system": "",
            "template": "",
            "options": {},
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let body = resp.text().await.unwrap();
    let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty());

    // Each line should be valid NDJSON with 'response' field
    for (i, line) in lines.iter().enumerate() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Line {i} not valid JSON: {e}\nLine: {line}"));
        assert!(parsed["model"].as_str().is_some());
        assert!(parsed["done"].is_boolean());
        if parsed["done"] == false {
            assert!(
                parsed["response"].is_string(),
                "Content line {i} missing 'response'"
            );
        }
    }
}
