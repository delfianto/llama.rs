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

    // Dynamic column width based on longest model name
    let name_width = models
        .iter()
        .map(|m| m.name.len())
        .max()
        .unwrap_or(4)
        .max(4); // at least "NAME" width

    println!(
        "{:<width$}  {:>10}  {:>14}",
        "NAME",
        "SIZE",
        "MODIFIED",
        width = name_width
    );

    for model in &models {
        println!(
            "{:<width$}  {:>10}  {:>14}",
            model.name,
            format_size(model.size),
            format_relative_time(model.modified),
            width = name_width
        );
    }

    Ok(())
}
