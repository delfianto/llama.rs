use crate::config::Config;
use crate::config::resolve::resolve_model_path;
use crate::error::output;
use crate::process::cli::spawn_cli;

/// Execute the `llama run` command — start interactive REPL via llama-cli.
#[allow(clippy::unused_async)]
pub async fn exec(config: &Config, model: &str) -> anyhow::Result<()> {
    let model_path = resolve_model_path(&config.models_dir, model)?;

    output::info(&format!("Model:        {}", model_path.display()));
    output::info("Mode:         interactive REPL");
    output::info(&format!("GPU layers:   {}", config.gpu_layers));
    if let Some(ref ts) = config.tensor_split {
        output::info(&format!("Tensor split: {ts}"));
    }
    output::info(&format!("Context size: {}", config.ctx_size));
    eprintln!();

    let mut handle = spawn_cli(config, &model_path)?;

    let status = handle.child.wait()?;

    if !status.success() {
        anyhow::bail!("llama-cli exited with status {status}");
    }
    Ok(())
}
