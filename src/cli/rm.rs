use std::time::Duration;

use crate::config::resolve::resolve_model_path;
use crate::config::Config;
use crate::error::output;
use crate::model::{cleanup_empty_dirs, find_process_using_model};

/// Execute the `llama rm` command — delete a downloaded model.
pub async fn exec(config: &Config, model: &str) -> anyhow::Result<()> {
    let model_path = resolve_model_path(&config.models_dir, model)?;

    match find_process_using_model(&model_path) {
        Ok(Some(pid)) => {
            output::warn(&format!("Model is in use by process {pid} — stopping it"));
            kill_process(pid);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        Ok(None) => {}
        Err(e) => {
            tracing::debug!("Could not check for running processes: {e}");
        }
    }

    tokio::fs::remove_file(&model_path).await?;
    output::success(&format!("Deleted: {}", model_path.display()));

    cleanup_empty_dirs(&model_path, &config.models_dir)?;

    Ok(())
}

fn kill_process(pid: u32) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        #[allow(clippy::cast_possible_wrap)]
        let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
    }
}
