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

    // Ignore SIGINT in the parent while the child is running.
    //
    // Why: both parent and child are in the same foreground process group.
    // When Ctrl+C is pressed, the kernel delivers SIGINT to EVERY process
    // in the group. The child (llama-cli) handles it:
    //   - Single Ctrl+C: cancels current generation, returns to REPL prompt
    //   - Double Ctrl+C: exits with code 130
    //
    // If we DON'T ignore SIGINT in the parent, tokio's default handler kills
    // the parent on the first Ctrl+C, orphaning the child.
    //
    // If we FORWARD SIGINT (the previous bug), the child gets TWO signals
    // (one from the kernel, one from us) and treats it as double Ctrl+C → exit.
    //
    // The shell script avoided all this by using `exec` (no parent process).
    // We can't exec, so we ignore SIGINT and let the child handle it naturally.
    #[cfg(unix)]
    let prev_handler = ignore_sigint();

    let status = handle.child.wait()?;

    // Restore default SIGINT handling
    #[cfg(unix)]
    restore_sigint(prev_handler);

    if !status.success() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            // Signal 2 = killed by SIGINT (shouldn't normally happen since
            // llama-cli handles SIGINT, but just in case)
            if status.signal() == Some(2) {
                return Ok(());
            }
            // Exit code 130 = llama-cli's normal double-Ctrl+C exit (128 + 2)
            if status.code() == Some(130) {
                return Ok(());
            }
        }
        anyhow::bail!("llama-cli exited with status {status}");
    }
    Ok(())
}

/// Set SIGINT to SIG_IGN (ignore). Returns the previous handler for restoration.
#[cfg(unix)]
#[allow(unsafe_code)]
fn ignore_sigint() -> nix::sys::signal::SigHandler {
    unsafe {
        nix::sys::signal::signal(
            nix::sys::signal::Signal::SIGINT,
            nix::sys::signal::SigHandler::SigIgn,
        )
        .unwrap_or(nix::sys::signal::SigHandler::SigDfl)
    }
}

/// Restore the previous SIGINT handler.
#[cfg(unix)]
#[allow(unsafe_code)]
fn restore_sigint(prev: nix::sys::signal::SigHandler) {
    unsafe {
        let _ = nix::sys::signal::signal(nix::sys::signal::Signal::SIGINT, prev);
    }
}
