//! Integration tests for the HuggingFace download manager using a real model.
//!
//! Downloads Mozilla/llama-test-model:tiny-llama (~27KB) — small enough
//! to run in CI without concern.
//!
//! Run with: cargo test --test live_download

use std::fs;

use tempfile::TempDir;

use llama_rs::config::Config;
use llama_rs::download::download_file;
use llama_rs::download::hf::{resolve_gguf_filename, ModelSpec};

const SPEC: &str = "Mozilla/llama-test-model:tiny-llama";
const EXPECTED_SIZE: u64 = 27488;
const GGUF_FILENAME: &str = "tiny-llama.gguf";

fn parse_spec() -> ModelSpec {
    ModelSpec::parse(SPEC).unwrap()
}

fn download_url() -> String {
    parse_spec().download_url(GGUF_FILENAME)
}

// ─── Basic Download ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_download_real_model() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = tmp.path().join(spec.filename());

    download_file(&download_url(), &dest, 1, None)
        .await
        .unwrap();

    assert!(dest.exists());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);
}

#[tokio::test]
async fn test_download_creates_org_structure() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = spec.local_path(tmp.path());

    assert_eq!(
        dest.file_name().unwrap(),
        "llama-test-model-tiny-llama.gguf"
    );
    assert_eq!(dest.parent().unwrap().file_name().unwrap(), "Mozilla");

    tokio::fs::create_dir_all(dest.parent().unwrap())
        .await
        .unwrap();
    download_file(&download_url(), &dest, 1, None)
        .await
        .unwrap();

    assert!(dest.exists());
    assert!(tmp.path().join("Mozilla").is_dir());
    assert!(!tmp.path().join("Mozilla/llama-test-model").is_dir());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);
}

#[tokio::test]
async fn test_download_parallel_connections() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = tmp.path().join(spec.filename());

    download_file(&download_url(), &dest, 4, None)
        .await
        .unwrap();

    assert!(dest.exists());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);
}

#[tokio::test]
async fn test_download_skip_existing() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = tmp.path().join(spec.filename());

    download_file(&download_url(), &dest, 1, None)
        .await
        .unwrap();
    let modified1 = fs::metadata(&dest).unwrap().modified().unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    assert!(dest.exists());
    let modified2 = fs::metadata(&dest).unwrap().modified().unwrap();
    assert_eq!(modified1, modified2);
}

// ─── Error Handling ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_download_nonexistent_file_404() {
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("nope.gguf");

    let url = "https://huggingface.co/Mozilla/llama-test-model/resolve/main/nonexistent.gguf";
    let result = download_file(url, &dest, 1, None).await;

    assert!(result.is_err());
    assert!(!dest.exists());
}

#[tokio::test]
async fn test_download_nonexistent_repo_404() {
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("nope.gguf");

    let url =
        "https://huggingface.co/nonexistent-org-xyz/nonexistent-repo-xyz/resolve/main/model.gguf";
    let result = download_file(url, &dest, 1, None).await;

    assert!(result.is_err());
    assert!(!dest.exists());
}

#[tokio::test]
async fn test_download_temp_file_cleanup() {
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("model.gguf");
    let temp_path = dest.with_extension("part");

    let url = "https://huggingface.co/Mozilla/llama-test-model/resolve/main/nonexistent.gguf";
    let _ = download_file(url, &dest, 1, None).await;

    assert!(!dest.exists());
    assert!(!temp_path.exists());
}

// ─── Full Lifecycle ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_pull_ls_rm_lifecycle() {
    let tmp = TempDir::new().unwrap();
    let config = Config::from_env();
    let spec = parse_spec();

    let dest = spec.local_path(tmp.path());
    assert!(!dest.exists());

    tokio::fs::create_dir_all(dest.parent().unwrap())
        .await
        .unwrap();
    download_file(
        &download_url(),
        &dest,
        config.download_connections,
        config.hf_token.as_deref(),
    )
    .await
    .unwrap();

    assert!(dest.exists());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);

    let models = llama_rs::model::scan_models(tmp.path()).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "Mozilla/llama-test-model-tiny-llama");
    assert_eq!(models[0].size, EXPECTED_SIZE);

    tokio::fs::remove_file(&dest).await.unwrap();
    llama_rs::model::cleanup_empty_dirs(&dest, tmp.path()).unwrap();
    assert!(!tmp.path().join("Mozilla").exists());
}

#[tokio::test]
async fn test_resolve_spec_after_pull() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();

    let dest = spec.local_path(tmp.path());
    tokio::fs::create_dir_all(dest.parent().unwrap())
        .await
        .unwrap();
    download_file(&download_url(), &dest, 1, None)
        .await
        .unwrap();

    // Resolve via spec syntax
    let resolved = llama_rs::config::resolve::resolve_model_path(tmp.path(), SPEC).unwrap();
    assert_eq!(resolved, dest);

    // Resolve via relative path
    let resolved = llama_rs::config::resolve::resolve_model_path(
        tmp.path(),
        "Mozilla/llama-test-model-tiny-llama.gguf",
    )
    .unwrap();
    assert_eq!(resolved, dest);

    // Resolve without .gguf extension
    let resolved = llama_rs::config::resolve::resolve_model_path(
        tmp.path(),
        "Mozilla/llama-test-model-tiny-llama",
    )
    .unwrap();
    assert_eq!(resolved, dest);
}

// ─── HuggingFace API Resolution ──────────────────────────────────────────────

#[tokio::test]
async fn test_resolve_gguf_filename() {
    let spec = parse_spec();
    let client = reqwest::Client::builder()
        .user_agent("llama-rs/0.1.0")
        .build()
        .unwrap();

    let filename = resolve_gguf_filename(&client, &spec).await.unwrap();
    assert_eq!(filename, GGUF_FILENAME);
}

#[tokio::test]
async fn test_resolve_gguf_filename_no_match() {
    let spec = ModelSpec::parse("Mozilla/llama-test-model:Q99_NONEXISTENT").unwrap();
    let client = reqwest::Client::builder()
        .user_agent("llama-rs/0.1.0")
        .build()
        .unwrap();

    let result = resolve_gguf_filename(&client, &spec).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No GGUF file matching") || err.contains("Available"),
        "Should list available files: {err}"
    );
}

#[tokio::test]
async fn test_resolve_gguf_filename_bad_repo() {
    let spec = ModelSpec::parse("nonexistent-org-xyz/nonexistent-repo-xyz:Q4_K_M").unwrap();
    let client = reqwest::Client::builder()
        .user_agent("llama-rs/0.1.0")
        .build()
        .unwrap();

    let result = resolve_gguf_filename(&client, &spec).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_hf_prefix_pull_flow() {
    let spec = ModelSpec::parse("hf.co/Mozilla/llama-test-model:tiny-llama").unwrap();
    assert_eq!(spec.org, "Mozilla");
    assert_eq!(spec.repo, "llama-test-model");

    let client = reqwest::Client::builder()
        .user_agent("llama-rs/0.1.0")
        .build()
        .unwrap();

    let filename = resolve_gguf_filename(&client, &spec).await.unwrap();
    assert_eq!(filename, GGUF_FILENAME);

    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join(spec.filename());
    let url = spec.download_url(&filename);

    download_file(&url, &dest, 1, None).await.unwrap();
    assert!(dest.exists());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);
}
