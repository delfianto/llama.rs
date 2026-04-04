use crate::config::Config;
use crate::model::types::parse_model_display;
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

    // Parse all models into display components
    let parsed: Vec<_> = models
        .iter()
        .map(|m| (parse_model_display(&m.name), m))
        .collect();

    // Compute dynamic column widths
    let org_w = parsed
        .iter()
        .map(|(d, _)| d.org.len())
        .max()
        .unwrap_or(3)
        .max(3); // "ORG"
    let model_w = parsed
        .iter()
        .map(|(d, _)| d.model.len())
        .max()
        .unwrap_or(5)
        .max(5); // "MODEL"
    let quant_w = parsed
        .iter()
        .map(|(d, _)| d.quant.len())
        .max()
        .unwrap_or(5)
        .max(5); // "QUANT"

    // Header
    println!(
        "{:<org_w$}  {:<model_w$}  {:<quant_w$}  {:>10}  {:>14}",
        "ORG", "MODEL", "QUANT", "SIZE", "MODIFIED",
    );

    // Rows
    for (display, model) in &parsed {
        println!(
            "{:<org_w$}  {:<model_w$}  {:<quant_w$}  {:>10}  {:>14}",
            display.org,
            display.model,
            display.quant,
            format_size(model.size),
            format_relative_time(model.modified),
        );
    }

    Ok(())
}
