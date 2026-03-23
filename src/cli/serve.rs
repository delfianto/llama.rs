use std::sync::Arc;
use std::time::Duration;

use tracing::info;

use crate::api::{self, AppState};
use crate::config::resolve::resolve_model_path;
use crate::config::Config;
use crate::error::output;
use crate::error::LlamaError;
use crate::process::health::wait_for_ready;
use crate::process::server::{shutdown_server, spawn_server};

/// Execute the `llama serve` command — start API server via llama-server.
pub async fn exec(config: &Config, model: &str) -> anyhow::Result<()> {
    let model_path = resolve_model_path(&config.models_dir, model)?;

    let model_name = model_path
        .file_name()
        .map_or_else(|| model.to_string(), |f| f.to_string_lossy().to_string());

    // Startup banner (colored, matching shell script style)
    output::info(&format!("Model:        {}", model_path.display()));
    output::info(&format!(
        "Endpoint:     http://{}:{}",
        config.host, config.port
    ));
    output::info(&format!("GPU layers:   {}", config.gpu_layers));
    if let Some(ref ts) = config.tensor_split {
        output::info(&format!("Tensor split: {ts}"));
    }
    output::info(&format!("Context size: {}", config.ctx_size));
    output::info(&format!(
        "Flash attn:   {}",
        if config.flash_attn { "on" } else { "off" }
    ));
    output::info(&format!(
        "Mlock:        {}",
        if config.mlock { "on" } else { "off" }
    ));
    eprintln!();

    let mut server_state = spawn_server(config, &model_path).await?;

    wait_for_ready(&server_state.internal_url, Duration::from_secs(120)).await?;
    output::success("Model loaded!");

    // Build axum proxy
    let app_state = AppState {
        config: Arc::new(Config::from_env()),
        llama_server_url: server_state.internal_url.clone(),
        model_name,
        http_client: reqwest::Client::new(),
    };

    let router = api::build_router(app_state);
    let bind_addr = format!("{}:{}", config.host, config.port);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::AddrInUse {
                LlamaError::PortInUse { port: config.port }.into()
            } else {
                anyhow::Error::from(e)
            }
        })?;

    output::success(&format!("Proxy server listening on http://{bind_addr}"));
    info!("Press Ctrl+C to stop");

    // Run axum with graceful shutdown on Ctrl+C
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to listen for Ctrl+C");
            output::info("Shutting down...");
        })
        .await?;

    shutdown_server(&mut server_state, Duration::from_secs(5)).await;
    output::success("Server stopped.");

    Ok(())
}
