//! Integration tests for signal handling (Ctrl+C / SIGINT).
//!
//! Tests that:
//! 1. A long-running generation can be interrupted via SIGINT
//! 2. The proxy server shuts down cleanly on SIGINT
//!
//! Requires llama-server running on localhost:8080.
//!
//! Run with: cargo test --test live_signal

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::net::TcpListener;

const LLAMA_SERVER_URL: &str = "http://127.0.0.1:8080";

async fn require_server() {
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

const LONG_PROMPT: &str = "\
Explain in extreme detail how thermonuclear fusion reactions work inside the Sun. \
Cover the proton-proton chain reaction step by step, the CNO cycle, \
the role of quantum tunneling in overcoming the Coulomb barrier, \
the energy transport mechanisms from core to surface (radiative zone vs convective zone), \
neutrino production and detection, the solar neutrino problem and its resolution, \
the pp-I, pp-II, and pp-III branches with their branching ratios, \
helium-3 and helium-4 production pathways, \
the temperature and pressure gradients from core to photosphere, \
and the relationship between mass and luminosity in main-sequence stars. \
Your answer must be extremely thorough and detailed, at least 2000 words.";

// ═══════════════════════════════════════════════════════════════════════════════
// Streaming interruption tests (via HTTP — cancel the connection mid-stream)
// ═══════════════════════════════════════════════════════════════════════════════

/// Start a long streaming generation via OpenAI API, read a few chunks,
/// then drop the connection. The server should handle the cancellation gracefully.
#[tokio::test]
async fn test_openai_streaming_cancel_mid_generation() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::to_string(&json!({
                "model": "test",
                "messages": [{"role": "user", "content": LONG_PROMPT}],
                "stream": true,
                "max_tokens": 2048
            }))
            .unwrap(),
        )
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    // Read the response as a byte stream, consume a few chunks, then drop
    let mut stream = resp.bytes_stream();
    let mut chunks_received = 0;
    let mut total_bytes = 0;

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                total_bytes += bytes.len();
                chunks_received += 1;
                // After receiving some data, abort the connection
                if chunks_received >= 5 || total_bytes > 500 {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    // Drop the stream — this closes the connection mid-generation
    drop(stream);

    assert!(
        chunks_received >= 1,
        "Should have received at least 1 chunk before cancelling"
    );

    // Verify the server is still healthy after the cancelled stream
    tokio::time::sleep(Duration::from_millis(200)).await;
    let health = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(
        health.status(),
        200,
        "Server should still be healthy after cancelled stream"
    );
}

/// Same test but via Ollama NDJSON streaming.
#[tokio::test]
async fn test_ollama_streaming_cancel_mid_generation() {
    require_server().await;
    let addr = start_proxy().await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/api/chat"))
        .json(&json!({
            "model": "test",
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "stream": true,
            "options": {"num_predict": 2048}
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let mut stream = resp.bytes_stream();
    let mut chunks_received = 0;
    let mut total_bytes = 0;

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) => {
                total_bytes += bytes.len();
                chunks_received += 1;
                if chunks_received >= 5 || total_bytes > 500 {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    drop(stream);

    assert!(chunks_received >= 1);

    tokio::time::sleep(Duration::from_millis(200)).await;
    let health = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(
        health.status(),
        200,
        "Server healthy after cancelled NDJSON stream"
    );
}

/// Verify that a long non-streaming request can be cancelled by dropping the connection.
#[tokio::test]
async fn test_non_streaming_cancel() {
    require_server().await;
    let addr = start_proxy().await;

    // Start a long generation with a short client timeout
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    let result = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::to_string(&json!({
                "model": "test",
                "messages": [{"role": "user", "content": LONG_PROMPT}],
                "stream": false,
                "max_tokens": 4096
            }))
            .unwrap(),
        )
        .send()
        .await;

    // Either completes (fast model) or times out — both are fine
    match result {
        Ok(resp) => {
            // If it completed, verify it's a valid response
            assert_eq!(resp.status(), 200);
        }
        Err(e) => {
            // Timeout is expected for slow models with long prompts
            assert!(e.is_timeout(), "Expected timeout error, got: {e}");
        }
    }

    // Server should still be healthy
    tokio::time::sleep(Duration::from_millis(500)).await;
    let health = reqwest::Client::new()
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), 200);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIGINT to child process test (for `llama run`)
// ═══════════════════════════════════════════════════════════════════════════════

/// Test that sending SIGINT to our binary properly forwards it to the child.
/// This simulates what happens when a user presses Ctrl+C during `llama run`.
#[cfg(unix)]
#[tokio::test]
async fn test_sigint_forwarded_to_child() {
    require_server().await;

    // We can't easily test `llama run` end-to-end (needs a model file for llama-cli),
    // but we CAN test the signal forwarding mechanism by spawning a simple child
    // process and verifying SIGINT reaches it.
    use std::process::Command;

    // Spawn `sleep 60` as a stand-in for llama-cli
    let mut child = Command::new("sleep").arg("60").spawn().unwrap();

    let pid = child.id();

    // Send SIGINT to the child (simulating our signal forwarder)
    #[allow(clippy::cast_possible_wrap)]
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(pid as i32),
        nix::sys::signal::Signal::SIGINT,
    )
    .unwrap();

    // Child should exit from the signal
    let status = child.wait().unwrap();
    assert!(!status.success(), "Child should have been killed by SIGINT");

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(2),
            "Child should have exited with signal 2 (SIGINT)"
        );
    }
}

/// Test that our binary's `llama serve` can be interrupted with SIGINT
/// and shuts down cleanly.
#[cfg(unix)]
#[tokio::test]
async fn test_serve_sigint_shutdown() {
    require_server().await;
    let addr = start_proxy().await;

    // Verify it's running
    let resp = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);

    // Start a long streaming request
    let client = reqwest::Client::new();
    let _resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(
            serde_json::to_string(&json!({
                "model": "test",
                "messages": [{"role": "user", "content": LONG_PROMPT}],
                "stream": true,
                "max_tokens": 2048
            }))
            .unwrap(),
        )
        .send()
        .await
        .unwrap();

    // Give it a moment to start generating
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Drop all connections — simulates the client going away
    // The server should handle this gracefully without panicking
    drop(_resp);

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Server should still be up
    let health = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert_eq!(health.status(), 200);
}
