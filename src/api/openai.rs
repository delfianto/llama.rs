use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tracing::debug;

use super::types::{ModelListResponse, ModelObject};
use super::AppState;

/// `POST /v1/chat/completions` — raw passthrough proxy to llama-server.
///
/// The request body is forwarded as-is without any deserialization or
/// transformation. The response (including status code, headers, and body)
/// is relayed back verbatim. This ensures full compatibility with any
/// OpenAI client — whatever the frontend sends, llama.cpp receives unchanged.
pub async fn chat_completions(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, Response> {
    let url = format!("{}/v1/chat/completions", state.llama_server_url);

    debug!("Proxying /v1/chat/completions to {url}");

    // Forward the raw body as-is
    let upstream_resp = state
        .http_client
        .post(&url)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("Upstream connection failed: {e}");
            (
                StatusCode::BAD_GATEWAY,
                format!("Upstream connection failed: {e}"),
            )
                .into_response()
        })?;

    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Copy all relevant response headers
    let mut headers = HeaderMap::new();
    for (name, value) in upstream_resp.headers() {
        // Skip hop-by-hop headers that shouldn't be proxied
        let skip = matches!(
            name.as_str(),
            "transfer-encoding" | "connection" | "keep-alive"
        );
        if !skip {
            headers.insert(name.clone(), value.clone());
        }
    }

    // Check if this is a streaming response (text/event-stream)
    let is_streaming = upstream_resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/event-stream"));

    if is_streaming {
        // Stream the response body as-is — no SSE parsing, no transformation
        let byte_stream = upstream_resp.bytes_stream();
        let body = Body::from_stream(byte_stream);

        // Ensure content-type is set for SSE
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );

        Ok((status, headers, body).into_response())
    } else {
        // Non-streaming: read full body and relay
        let response_body = upstream_resp.bytes().await.map_err(|e| {
            tracing::error!("Failed to read upstream response: {e}");
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to read response: {e}"),
            )
                .into_response()
        })?;

        Ok((status, headers, response_body).into_response())
    }
}

/// `GET /v1/models` — return the currently loaded model.
pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model",
            owned_by: "local",
        }],
    })
}
