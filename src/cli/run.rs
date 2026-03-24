use std::sync::atomic::{AtomicU32, Ordering};

use crate::config::Config;
use crate::config::resolve::resolve_model_path;
use crate::error::output;
use crate::process::cli::spawn_cli;

/// Global child PID for signal forwarding.
static CHILD_PID: AtomicU32 = AtomicU32::new(0);

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
    CHILD_PID.store(handle.pid, Ordering::SeqCst);

    // Install signal forwarder: Ctrl+C sends SIGINT to the child instead of
    // killing our parent. This lets llama-cli cancel generation gracefully.
    #[cfg(unix)]
    install_signal_forwarder();

    let status = handle.child.wait()?;
    CHILD_PID.store(0, Ordering::SeqCst);

    if !status.success() {
        // SIGINT exit (signal 2) is normal — user pressed Ctrl+C
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            if status.signal() == Some(2) {
                return Ok(());
            }
        }
        anyhow::bail!("llama-cli exited with status {status}");
    }
    Ok(())
}

/// Install a raw SIGINT handler that forwards the signal to the child process.
///
/// The shell script used `exec` to replace itself with llama-cli, so Ctrl+C
/// went directly to it. We can't do that (we need cleanup), so instead we
/// catch SIGINT and forward it to the child PID.
#[cfg(unix)]
#[allow(unsafe_code)]
fn install_signal_forwarder() {
    unsafe {
        nix::sys::signal::signal(
            nix::sys::signal::Signal::SIGINT,
            nix::sys::signal::SigHandler::Handler(sigint_handler),
        )
        .ok();
    }
}

#[cfg(unix)]
extern "C" fn sigint_handler(_sig: std::ffi::c_int) {
    let pid = CHILD_PID.load(Ordering::SeqCst);
    if pid != 0 {
        #[allow(clippy::cast_possible_wrap)]
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGINT,
        );
    }
}
