use crate::config::Config;
use crate::download::download_file;
use crate::download::hf::{resolve_gguf_filename, ModelSpec};
use crate::error::output;

/// Execute the `llama pull` command — download a GGUF model from HuggingFace.
///
/// Spec format: `[hf.co/]org/repo:quant`
///
/// The quant tag is matched against filenames in the repo via the HuggingFace API,
/// since GGUF filenames are arbitrary and don't follow a fixed convention.
pub async fn exec(config: &Config, spec: &str) -> anyhow::Result<()> {
    let model = ModelSpec::parse(spec).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid model spec: {spec}\n  \
             Expected format: org/repo:quant\n  \
             Example: mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M"
        )
    })?;

    let dest = model.local_path(&config.models_dir);

    if dest.exists() {
        output::info(&format!("Model already exists: {}", model.display_name()));
        return Ok(());
    }

    // Resolve the actual GGUF filename in the repo
    output::info(&format!(
        "Resolving {} in {}...",
        model.quant,
        model.repo_id()
    ));

    let client = reqwest::Client::builder()
        .user_agent("llama-rs/0.1.0")
        .build()?;

    let gguf_filename = resolve_gguf_filename(&client, &model).await?;
    let url = model.download_url(&gguf_filename);

    output::info(&format!("Pulling {}", model.display_name()));
    output::info(&format!("File:   {gguf_filename}"));
    output::info(&format!("From:   {url}"));
    output::info(&format!("To:     {}", dest.display()));

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    download_file(
        &url,
        &dest,
        config.download_connections,
        config.hf_token.as_deref(),
    )
    .await?;

    output::success(&format!("Pull complete: {}", model.display_name()));
    Ok(())
}
