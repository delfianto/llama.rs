use crate::config::Config;
use crate::model::{format_relative_time, format_size, scan_models};

/// Execute the `llama ls` command — list downloaded models.
#[allow(clippy::unused_async)]
pub async fn exec(config: &Config) -> anyhow::Result<()> {
    let models = scan_models(&config.models_dir)?;

    if models.is_empty() {
        println!("No models found in {}", config.models_dir.display());
        println!("Use 'llama pull <org/repo:quant>' to download a model.");
        return Ok(());
    }

    println!("{:<55} {:>10} {:>14}", "NAME", "SIZE", "MODIFIED");

    for model in &models {
        println!(
            "{:<55} {:>10} {:>14}",
            model.name,
            format_size(model.size),
            format_relative_time(model.modified),
        );
    }

    Ok(())
}
