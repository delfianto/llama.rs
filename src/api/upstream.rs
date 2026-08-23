use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use tracing::debug;

use super::AppState;

/// Relay an otherwise-unhandled request to the internal llama-server.
///
/// This exposes llama.cpp's built-in Web UI at `/` together with its static
/// assets and llama.cpp-specific API endpoints, while the wrapper's explicit
/// OpenAI and Ollama routes continue to take precedence.
#[allow(clippy::result_large_err)]
pub async fn proxy(
    State(state): State<AppState>,
    request: Request<Body>,
) -> Result<Response, Response> {
    let (parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map_or("/", axum::http::uri::PathAndQuery::as_str);
    let url = format!("{}{path_and_query}", state.llama_server_url);

    debug!("Proxying {} {path_and_query} to llama-server", parts.method);

    let mut request_headers = parts.headers;
    remove_hop_by_hop_headers(&mut request_headers);
    request_headers.remove(header::HOST);

    let upstream_response = state
        .http_client
        .request(parts.method, url)
        .headers(request_headers)
        .body(reqwest::Body::wrap_stream(body.into_data_stream()))
        .send()
        .await
        .map_err(|error| {
            tracing::error!("Upstream connection failed: {error}");
            (
                StatusCode::BAD_GATEWAY,
                format!("Upstream connection failed: {error}"),
            )
                .into_response()
        })?;

    let status = upstream_response.status();
    let mut response_headers = upstream_response.headers().clone();
    remove_hop_by_hop_headers(&mut response_headers);
    let response_body = Body::from_stream(upstream_response.bytes_stream());

    Ok((status, response_headers, response_body).into_response())
}

fn remove_hop_by_hop_headers(headers: &mut HeaderMap) {
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }
}
