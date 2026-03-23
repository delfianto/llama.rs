use std::time::Duration;

use tracing::{debug, info};

use crate::error::LlamaError;

/// Poll llama-server's `/health` endpoint until it returns 200.
///
/// - Polls every `interval` (default 500ms)
/// - Gives up after `timeout` (default 120s for large models)
pub async fn wait_for_ready(base_url: &str, timeout: Duration) -> anyhow::Result<()> {
    let interval = Duration::from_millis(500);
    let health_url = format!("{base_url}/health");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    info!("Waiting for llama-server to load model...");

    let deadline = tokio::time::Instant::now() + timeout;
    let mut attempts = 0u32;

    loop {
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(LlamaError::HealthTimeout {
                seconds: timeout.as_secs(),
            });
        }

        attempts += 1;
        match client.get(&health_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!("llama-server is ready (took {attempts} health checks)");
                return Ok(());
            }
            Ok(resp) => {
                debug!("Health check attempt {attempts}: status {}", resp.status());
            }
            Err(e) => {
                debug!("Health check attempt {attempts}: {e}");
            }
        }

        tokio::time::sleep(interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_health_check_succeeds_on_200() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let result = wait_for_ready(&mock_server.uri(), Duration::from_secs(5)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_retries_then_succeeds() {
        let mock_server = MockServer::start().await;

        // First 2 requests return 503, then 200
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&mock_server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let result = wait_for_ready(&mock_server.uri(), Duration::from_secs(10)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_times_out() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&mock_server)
            .await;

        let result = wait_for_ready(&mock_server.uri(), Duration::from_secs(2)).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timed out"));
    }

    #[tokio::test]
    async fn test_health_check_connection_refused_retries() {
        // Use a port that nothing is listening on
        let result = wait_for_ready("http://127.0.0.1:1", Duration::from_secs(2)).await;
        assert!(result.is_err());
    }
}
