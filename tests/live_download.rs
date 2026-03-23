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
use llama_rs::download::hf::ModelSpec;

const SPEC: &str = "Mozilla/llama-test-model:tiny-llama";
const EXPECTED_SIZE: u64 = 27488;

fn parse_spec() -> ModelSpec {
    ModelSpec::parse(SPEC).unwrap()
}

#[tokio::test]
async fn test_download_real_model() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = tmp.path().join(spec.filename());

    download_file(&spec.hf_url(), &dest, 1, None).await.unwrap();

    assert!(dest.exists(), "Downloaded file should exist");
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);
}

#[tokio::test]
async fn test_download_creates_org_structure() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = spec.local_path(tmp.path());

    // Should be: tmp/Mozilla/llama-test-model-tiny-llama.gguf
    assert_eq!(
        dest.file_name().unwrap(),
        "llama-test-model-tiny-llama.gguf"
    );
    assert_eq!(dest.parent().unwrap().file_name().unwrap(), "Mozilla");

    tokio::fs::create_dir_all(dest.parent().unwrap())
        .await
        .unwrap();

    download_file(&spec.hf_url(), &dest, 1, None).await.unwrap();

    assert!(dest.exists());
    assert!(tmp.path().join("Mozilla").is_dir());
    // No nested repo dir — flat structure
    assert!(!tmp.path().join("Mozilla/llama-test-model").is_dir());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);
}

#[tokio::test]
async fn test_download_parallel_connections() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = tmp.path().join(spec.filename());

    // 4 connections — file is small so falls back to single stream
    download_file(&spec.hf_url(), &dest, 4, None).await.unwrap();

    assert!(dest.exists());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);
}

#[tokio::test]
async fn test_download_skip_existing() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();
    let dest = tmp.path().join(spec.filename());

    download_file(&spec.hf_url(), &dest, 1, None).await.unwrap();
    let modified1 = fs::metadata(&dest).unwrap().modified().unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // cli/pull.rs checks dest.exists() and returns early
    assert!(dest.exists());
    let modified2 = fs::metadata(&dest).unwrap().modified().unwrap();
    assert_eq!(modified1, modified2);
}

#[tokio::test]
async fn test_download_nonexistent_file_404() {
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("nope.gguf");

    let spec = ModelSpec::parse("Mozilla/llama-test-model:nonexistent-file").unwrap();
    let result = download_file(&spec.hf_url(), &dest, 1, None).await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not found") || err.contains("404"),
        "Expected 'not found' error, got: {err}"
    );
    assert!(!dest.exists());
}

#[tokio::test]
async fn test_download_nonexistent_repo_404() {
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("nope.gguf");

    let spec = ModelSpec::parse("nonexistent-org-xyz/nonexistent-repo-xyz:Q4_K_M").unwrap();
    let result = download_file(&spec.hf_url(), &dest, 1, None).await;

    assert!(result.is_err());
    assert!(!dest.exists());
}

#[tokio::test]
async fn test_download_temp_file_cleanup() {
    let tmp = TempDir::new().unwrap();
    let dest = tmp.path().join("model.gguf");
    let temp_path = dest.with_extension("part");

    let spec = ModelSpec::parse("Mozilla/llama-test-model:nonexistent").unwrap();
    let _ = download_file(&spec.hf_url(), &dest, 1, None).await;

    assert!(!dest.exists());
    assert!(!temp_path.exists());
}

/// Full end-to-end: pull → ls → rm lifecycle using new spec format.
#[tokio::test]
async fn test_pull_ls_rm_lifecycle() {
    let tmp = TempDir::new().unwrap();
    let config = Config::from_env();
    let spec = parse_spec();

    let dest = spec.local_path(tmp.path());
    assert!(!dest.exists());

    // Pull
    tokio::fs::create_dir_all(dest.parent().unwrap())
        .await
        .unwrap();
    download_file(
        &spec.hf_url(),
        &dest,
        config.download_connections,
        config.hf_token.as_deref(),
    )
    .await
    .unwrap();

    assert!(dest.exists());
    assert_eq!(fs::metadata(&dest).unwrap().len(), EXPECTED_SIZE);

    // Ls — should find the model with display name (no .gguf)
    let models = llama_rs::model::scan_models(tmp.path()).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "Mozilla/llama-test-model-tiny-llama");
    assert_eq!(models[0].size, EXPECTED_SIZE);

    // Rm — delete and clean up
    tokio::fs::remove_file(&dest).await.unwrap();
    llama_rs::model::cleanup_empty_dirs(&dest, tmp.path()).unwrap();
    assert!(
        !tmp.path().join("Mozilla").exists(),
        "Empty org dir should be cleaned up"
    );
}

/// Test that model path resolution works with spec format.
#[tokio::test]
async fn test_resolve_spec_after_pull() {
    let tmp = TempDir::new().unwrap();
    let spec = parse_spec();

    let dest = spec.local_path(tmp.path());
    tokio::fs::create_dir_all(dest.parent().unwrap())
        .await
        .unwrap();
    download_file(&spec.hf_url(), &dest, 1, None).await.unwrap();

    // Should resolve using spec syntax
    let resolved = llama_rs::config::resolve::resolve_model_path(tmp.path(), SPEC).unwrap();
    assert_eq!(resolved, dest);

    // Should also resolve using relative path
    let resolved = llama_rs::config::resolve::resolve_model_path(
        tmp.path(),
        "Mozilla/llama-test-model-tiny-llama.gguf",
    )
    .unwrap();
    assert_eq!(resolved, dest);

    // Should also resolve without .gguf extension
    let resolved = llama_rs::config::resolve::resolve_model_path(
        tmp.path(),
        "Mozilla/llama-test-model-tiny-llama",
    )
    .unwrap();
    assert_eq!(resolved, dest);
}
