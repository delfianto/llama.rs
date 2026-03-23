pub mod ollama;
pub mod openai;
pub mod stream;
pub mod types;

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, head, post};
use tower_http::cors::CorsLayer;

use crate::config::Config;

/// Shared state available to all API handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    /// Base URL of the internal llama-server (e.g., `http://127.0.0.1:54321`).
    pub llama_server_url: String,
    /// Model name to report in `/v1/models` and Ollama responses.
    pub model_name: String,
    /// Shared HTTP client for proxying requests.
    pub http_client: reqwest::Client,
}

/// Build the axum router with all API routes.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Root — ollama CLI does HEAD / as connectivity check
        .route("/", head(root_head).get(root_get))
        // OpenAI-compatible
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::list_models))
        // Ollama-compatible
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/api/tags", get(ollama::tags))
        .route("/api/show", post(ollama::show))
        .route("/api/version", get(ollama::version))
        // Health
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health(State(_state): State<AppState>) -> impl IntoResponse {
    StatusCode::OK
}

/// `HEAD /` — ollama CLI connectivity check.
async fn root_head() -> impl IntoResponse {
    StatusCode::OK
}

/// `GET /` — ollama CLI might also GET /.
async fn root_get() -> &'static str {
    "llama.rs is running"
}
