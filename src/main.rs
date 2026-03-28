use std::sync::Arc;

use clap::{Parser, Subcommand};
use llama_rs::{cli, config, error::output};
use mimalloc::MiMalloc;
use tracing_subscriber::EnvFilter;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const ENV_HELP: &str = "\x1b[1mEnvironment variables:\x1b[0m
  LLAMA_MODELS_DIR       Model storage directory       (default: OS data dir)
  LLAMA_BIN_DIR          llama.cpp binary directory    (default: search PATH)
  LLAMA_GPU_LAYERS       Layers to offload to GPU      (default: 999 = all)
  LLAMA_CTX_SIZE         Context window size           (default: 32768)
  LLAMA_BATCH_SIZE       Batch size                    (default: 2048)
  LLAMA_THREADS          CPU threads                   (default: auto-detect)
  LLAMA_TENSOR_SPLIT     VRAM ratio per GPU            (e.g., 14,12)
  LLAMA_MAIN_GPU         Primary GPU index             (default: 0)
  LLAMA_FLASH_ATTN       Flash attention 1=on 0=off    (default: 1)
  LLAMA_MLOCK            Lock model in RAM             (default: 1)
  LLAMA_HOST             Server bind address           (default: 127.0.0.1)
  LLAMA_PORT             Server port                   (default: 8080)
  LLAMA_SYSTEM_PROMPT    System prompt for REPL        (default: You are a helpful assistant.)
  LLAMA_SYSTEM_PROMPT_FILE  Path to system prompt file (overrides LLAMA_SYSTEM_PROMPT)
  LLAMA_PROMPT_TEMPLATE_FILE  Path to chat template file (overrides LLAMA_PROMPT_TEMPLATE)
  LLAMA_PROMPT_TEMPLATE  Chat template string
  LLAMA_TEMPERATURE      Sampling temperature
  LLAMA_MAX_TOKENS       Max response tokens
  LLAMA_CTX_OVERFLOW     Context overflow: shift|stop  (default: shift)
  LLAMA_STOP             Stop strings, comma-separated (llama run only)
  LLAMA_TOP_K            Top-k sampling
  LLAMA_REPEAT_PENALTY   Repeat penalty
  LLAMA_PRESENCE_PENALTY Presence penalty
  LLAMA_TOP_P            Top-p / nucleus sampling
  LLAMA_MIN_P            Min-p sampling
  LLAMA_DOWNLOAD_CONNECTIONS  Parallel downloads       (default: 4)
  HF_TOKEN               HuggingFace token for gated models
  LLAMA_LOG              Log level                     (default: info)";

#[derive(Parser)]
#[command(
    name = "llama",
    version,
    about = "Ollama-like CLI wrapper for llama.cpp",
    after_help = ENV_HELP,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive REPL with a model
    Run {
        /// Model spec or path (e.g., org/repo:quant, org/file.gguf, /absolute/path.gguf)
        model: String,
    },
    /// Start API server with a model
    Serve {
        /// Model spec or path (e.g., org/repo:quant, org/file.gguf, /absolute/path.gguf)
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
