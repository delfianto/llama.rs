use crate::config::Config;
use crate::download::download_file;
use crate::download::hf::ModelSpec;
use crate::error::output;

/// Execute the `llama pull` command — download a GGUF model from HuggingFace.
///
/// Spec format: `org/repo:quant` (e.g., `mradermacher/Qwen3.5-27B-...-GGUF:Q4_K_M`)
pub async fn exec(config: &Config, spec: &str) -> anyhow::Result<()> {
    let model = ModelSpec::parse(spec).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid model spec: {spec}\n  \
             Expected format: org/repo:quant\n  \
             Example: mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M"
        )
    })?;

    let url = model.hf_url();
    let dest = model.local_path(&config.models_dir);

    if dest.exists() {
        output::info(&format!("Model already exists: {}", model.display_name()));
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    output::info(&format!("Pulling {}", model.display_name()));
    output::info(&format!("From:   {url}"));
    output::info(&format!("To:     {}", dest.display()));

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
