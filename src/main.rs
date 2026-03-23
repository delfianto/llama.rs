use std::sync::Arc;

use clap::{Parser, Subcommand};
use llama_rs::{cli, config, error::output};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "llama",
    version,
    about = "Ollama-like CLI wrapper for llama.cpp"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive REPL with a model
    Run {
        /// Model file (filename, relative path, or absolute path)
        model: String,
    },
    /// Start API server with a model
    Serve {
        /// Model file (filename, relative path, or absolute path)
        model: String,
    },
    /// Download a GGUF model from HuggingFace
    #[allow(clippy::doc_markdown)]
    Pull {
        /// Model spec (e.g., mradermacher/Qwen3.5-27B-...-GGUF:Q4_K_M)
        #[allow(clippy::doc_markdown)]
        spec: String,
    },
    /// List downloaded models
    Ls,
    /// Delete a downloaded model
    Rm {
        /// Model to delete
        model: String,
    },
}

fn init_logging() {
    let filter = EnvFilter::try_from_env("RUST_LOG")
        .or_else(|_| {
            EnvFilter::try_new(std::env::var("LLAMA_LOG").unwrap_or_else(|_| "info".to_string()))
        })
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

#[tokio::main]
async fn main() {
    init_logging();

    let config = Arc::new(config::Config::from_env());
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Run { model } => cli::run::exec(&config, &model).await,
        Commands::Serve { model } => cli::serve::exec(&config, &model).await,
        Commands::Pull { spec } => cli::pull::exec(&config, &spec).await,
        Commands::Ls => cli::ls::exec(&config).await,
        Commands::Rm { model } => cli::rm::exec(&config, &model).await,
    };

    if let Err(e) = result {
        output::error(&format!("{e:#}"));
        std::process::exit(1);
    }
}
