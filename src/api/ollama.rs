use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tracing::debug;

use super::stream::ndjson::{sse_to_ndjson_chat_stream, sse_to_ndjson_generate_stream};
use super::types::{
    ollama_chat_to_openai, ollama_generate_to_openai, openai_to_ollama_chat,
    openai_to_ollama_generate, OllamaChatRequest, OllamaGenerateRequest, OllamaModelInfo,
    OllamaShowRequest, OllamaShowResponse, OllamaTagsResponse,
};
use super::AppState;

/// `POST /api/chat` — Ollama-compatible chat endpoint.
pub async fn chat(
    State(state): State<AppState>,
    Json(request): Json<OllamaChatRequest>,
) -> Result<Response, StatusCode> {
    if request.stream {
        return stream_chat(state, &request).await;
    }

    let openai_body = ollama_chat_to_openai(&request, false);
    let url = format!("{}/v1/chat/completions", state.llama_server_url);

    debug!("Ollama /api/chat (non-streaming) → {url}");

    let upstream_resp = state
        .http_client
        .post(&url)
        .json(&openai_body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("Upstream request failed: {e}");
            StatusCode::BAD_GATEWAY
        })?;

    if !upstream_resp.status().is_success() {
        return Err(StatusCode::from_u16(upstream_resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR));
    }

    let body: serde_json::Value = upstream_resp.json().await.map_err(|e| {
        tracing::error!("Failed to parse upstream response: {e}");
        StatusCode::BAD_GATEWAY
    })?;

    let ollama_resp = openai_to_ollama_chat(&body, &state.model_name);
    Ok(Json(ollama_resp).into_response())
}

/// Stream chat response as NDJSON.
async fn stream_chat(state: AppState, request: &OllamaChatRequest) -> Result<Response, StatusCode> {
    let openai_body = ollama_chat_to_openai(request, true);
    let url = format!("{}/v1/chat/completions", state.llama_server_url);

    debug!("Ollama /api/chat (streaming NDJSON) → {url}");

    let upstream = state
        .http_client
        .post(&url)
        .json(&openai_body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("Upstream stream connect failed: {e}");
            StatusCode::BAD_GATEWAY
        })?;

    if !upstream.status().is_success() {
        return Err(StatusCode::from_u16(upstream.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR));
    }

    let ndjson_stream = sse_to_ndjson_chat_stream(upstream.bytes_stream(), &state.model_name);

    Ok(Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .header("Transfer-Encoding", "chunked")
        .body(Body::from_stream(ndjson_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

/// `POST /api/generate` — Ollama-compatible generate endpoint.
pub async fn generate(
    State(state): State<AppState>,
    Json(request): Json<OllamaGenerateRequest>,
) -> Result<Response, StatusCode> {
    if request.stream {
        return stream_generate(state, &request).await;
    }

    let openai_body = ollama_generate_to_openai(&request, false);
    let url = format!("{}/v1/chat/completions", state.llama_server_url);

    debug!("Ollama /api/generate (non-streaming) → {url}");

    let upstream_resp = state
        .http_client
        .post(&url)
        .json(&openai_body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("Upstream request failed: {e}");
            StatusCode::BAD_GATEWAY
        })?;

    if !upstream_resp.status().is_success() {
        return Err(StatusCode::from_u16(upstream_resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR));
    }

    let body: serde_json::Value = upstream_resp.json().await.map_err(|e| {
        tracing::error!("Failed to parse upstream response: {e}");
        StatusCode::BAD_GATEWAY
    })?;

    let ollama_resp = openai_to_ollama_generate(&body, &state.model_name);
    Ok(Json(ollama_resp).into_response())
}

/// Stream generate response as NDJSON.
async fn stream_generate(
    state: AppState,
    request: &OllamaGenerateRequest,
) -> Result<Response, StatusCode> {
    let openai_body = ollama_generate_to_openai(request, true);
    let url = format!("{}/v1/chat/completions", state.llama_server_url);

    debug!("Ollama /api/generate (streaming NDJSON) → {url}");

    let upstream = state
        .http_client
        .post(&url)
        .json(&openai_body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("Upstream stream connect failed: {e}");
            StatusCode::BAD_GATEWAY
        })?;

    if !upstream.status().is_success() {
        return Err(StatusCode::from_u16(upstream.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR));
    }

    let ndjson_stream = sse_to_ndjson_generate_stream(upstream.bytes_stream(), &state.model_name);

    Ok(Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .header("Transfer-Encoding", "chunked")
        .body(Body::from_stream(ndjson_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()))
}

/// `GET /api/tags` — Ollama-compatible model list.
pub async fn tags(State(state): State<AppState>) -> Json<OllamaTagsResponse> {
    let name = format!("{}:latest", state.model_name);
    Json(OllamaTagsResponse {
        models: vec![OllamaModelInfo {
            name: name.clone(),
            model: name,
            modified_at: chrono::Utc::now().to_rfc3339(),
            size: 0,
            digest: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            details: serde_json::json!({
                "parent_model": "",
                "format": "gguf",
                "family": "",
                "families": [""],
                "parameter_size": "",
                "quantization_level": ""
            }),
        }],
    })
}

/// `POST /api/show` — Ollama-compatible model info.
pub async fn show(
    State(state): State<AppState>,
    Json(_request): Json<OllamaShowRequest>,
) -> Json<OllamaShowResponse> {
    Json(OllamaShowResponse {
        modelfile: format!("FROM {}", state.model_name),
        parameters: String::new(),
        template: String::new(),
        model_info: serde_json::json!({
            "general.architecture": "llama",
            "general.type": "model",
            "general.quantization_version": 2
        }),
        details: serde_json::json!({
            "format": "gguf",
            "quantization_level": ""
        }),
    })
}

/// `GET /api/version` — Ollama version endpoint.
pub async fn version() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": "0.18.0"
    }))
}
