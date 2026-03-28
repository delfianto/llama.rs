use crate::config::Config;
use crate::download::download_file;
use crate::download::hf::{ModelSpec, resolve_gguf_filename};
use crate::error::output;

/// Execute the `llama pull` command — download a GGUF model from HuggingFace.
///
/// Spec format: `[hf.co/]org/repo:quant`
///
/// The quant tag is matched against filenames in the repo via the HuggingFace API,
/// since GGUF filenames are arbitrary and don't follow a fixed convention.
/// Also downloads metadata files (config.json, tokenizer files, etc.) when present.
pub async fn exec(config: &Config, spec: &str) -> anyhow::Result<()> {
    let model = ModelSpec::parse(spec).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid model spec: {spec}\n  \
             Expected format: org/repo:quant\n  \
             Example: mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF:Q4_K_M"
        )
    })?;

    // Resolve the actual GGUF filename and metadata files in the repo
    output::info(&format!(
        "Resolving {} in {}...",
        model.quant,
        model.repo_id()
    ));

    let client = reqwest::Client::builder()
        .user_agent("llama-rs/0.1.0")
        .build()?;

    let resolved = resolve_gguf_filename(&client, &model).await?;
    let dest = model.local_path(&config.models_dir, &resolved.gguf_filename);
    let url = model.download_url(&resolved.gguf_filename);

    if dest.exists() {
        eprint!("File already exists: {}. Override? [y/N] ", dest.display());
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !answer.trim().eq_ignore_ascii_case("y") {
            output::info("Skipped.");
            return Ok(());
        }
    }

    output::info(&format!("Pulling {}", model.display_name()));
    output::info(&format!("File:   {}", resolved.gguf_filename));
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

    // Download metadata files (config.json, tokenizer files, etc.)
    let repo_dir = model.local_dir(&config.models_dir);
    for filename in &resolved.metadata_files {
        let meta_dest = repo_dir.join(filename);
        if meta_dest.exists() {
            continue;
        }
        let meta_url = model.download_url(filename);
        tracing::debug!("Downloading metadata: {filename}");
        if let Err(e) = download_file(&meta_url, &meta_dest, 1, config.hf_token.as_deref()).await {
            tracing::warn!("Failed to download {filename}: {e}");
        }
    }

    output::success(&format!("Pull complete: {}", model.display_name()));
    Ok(())
}
