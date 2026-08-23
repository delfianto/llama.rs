#![allow(clippy::unwrap_used)]
//! Integration test — panicking on unexpected setup/response failures is expected here.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_help_flag() {
    Command::cargo_bin("llama")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Ollama-like CLI wrapper"));
}

#[test]
fn test_version_flag() {
    Command::cargo_bin("llama")
        .unwrap()
        .arg("--version")
        .assert()
        .success();
}

#[test]
fn test_no_args_shows_error() {
    Command::cargo_bin("llama").unwrap().assert().failure();
}

#[test]
fn test_run_subcommand_in_help() {
    Command::cargo_bin("llama")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("serve"))
        .stdout(predicate::str::contains("pull"))
        .stdout(predicate::str::contains("ls"))
        .stdout(predicate::str::contains("rm"));
}

// ─── Subcommand help tests ───────────────────────────────────────────────────

#[test]
fn test_run_help() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("interactive REPL"))
        .stdout(predicate::str::contains("--config"));
}

#[test]
fn test_serve_help() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("API server"))
        .stdout(predicate::str::contains("--device"));
}

#[test]
fn test_run_help_shows_device_option() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--device"))
        .stdout(predicate::str::contains("gpu0"));
}

#[test]
fn test_run_without_model_or_config_is_helpful() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["run"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("provide <MODEL>"));
}

#[test]
fn test_invalid_yaml_config_is_helpful() {
    let config = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(config.path(), "compute:\n  gpu_layerz: 99\n").unwrap();

    Command::cargo_bin("llama")
        .unwrap()
        .args(["serve", "--config", config.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid YAML config"))
        .stderr(predicate::str::contains("unknown field"));
}

#[cfg(unix)]
#[test]
fn test_run_uses_model_and_arguments_from_yaml() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::TempDir::new().unwrap();
    let model = tmp.path().join("model.gguf");
    let binary = tmp.path().join("llama-cli");
    let captured = tmp.path().join("arguments.txt");
    let profile = tmp.path().join("experiment.yaml");

    std::fs::write(&model, b"test model").unwrap();
    std::fs::write(
        &binary,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$LLAMA_CAPTURE_ARGS\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&binary, permissions).unwrap();

    std::fs::write(
        &profile,
        format!(
            "model: {}\npaths:\n  bin_dir: {}\ncompute:\n  device: gpu1\ninference:\n  context_size: 4096\nextra_args:\n  - --cache-type-k\n  - q8_0\n",
            model.display(),
            tmp.path().display()
        ),
    )
    .unwrap();

    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_CAPTURE_ARGS", &captured)
        .args(["run", "--config", profile.to_str().unwrap()])
        .assert()
        .success();

    let arguments = std::fs::read_to_string(captured).unwrap();
    assert!(arguments.contains(&format!("-m\n{}", model.display())));
    assert!(arguments.contains("--device\nCUDA1"));
    assert!(arguments.contains("-c\n4096"));
    assert!(arguments.contains("--cache-type-k\nq8_0"));
}

#[test]
fn test_pull_help() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["pull", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("HuggingFace").or(predicate::str::contains("GGUF")));
}

#[test]
fn test_ls_help() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["ls", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("List downloaded models"));
}

#[test]
fn test_rm_help() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["rm", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Delete"));
}

// ─── Model listing tests ────────────────────────────────────────────────────

#[test]
fn test_ls_shows_nested_models() {
    let tmp = tempfile::TempDir::new().unwrap();
    let repo_dir = tmp.path().join("org").join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    std::fs::write(repo_dir.join("repo-Q4_K_M.gguf"), vec![0u8; 2048]).unwrap();

    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", tmp.path().to_str().unwrap())
        .args(["ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains("org"))
        .stdout(predicate::str::contains("repo"))
        .stdout(predicate::str::contains("Q4_K_M"));
}

#[test]
fn test_ls_shows_size_and_modified() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("test.gguf"), vec![0u8; 1024]).unwrap();

    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", tmp.path().to_str().unwrap())
        .args(["ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODEL"))
        .stdout(predicate::str::contains("SIZE"))
        .stdout(predicate::str::contains("MODIFIED"));
}

// ─── Error message tests ────────────────────────────────────────────────────

#[test]
fn test_run_missing_model_shows_helpful_error() {
    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", "/tmp/llama_test_empty_nonexistent")
        .args(["run", "nonexistent.gguf"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error:"));
}

#[test]
fn test_serve_missing_binary_shows_helpful_error() {
    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_BIN_DIR", "/nonexistent/bin")
        .env("LLAMA_MODELS_DIR", "/tmp")
        .args(["serve", "/dev/null"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn test_rm_missing_model_shows_error() {
    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", "/tmp/llama_test_empty_nonexistent")
        .args(["rm", "nonexistent.gguf"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Error:"));
}

#[test]
fn test_ls_empty_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", tmp.path().to_str().unwrap())
        .args(["ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No models found"));
}

#[test]
fn test_custom_models_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let repo = tmp.path().join("myorg").join("myrepo");
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(repo.join("mymodel-Q4_K_M.gguf"), vec![0u8; 512]).unwrap();

    // LLAMA_MODELS_DIR should control where models are found
    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", tmp.path().to_str().unwrap())
        .args(["ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains("myorg"))
        .stdout(predicate::str::contains("myrepo"))
        .stdout(predicate::str::contains("Q4_K_M"));
}

// ─── New env var help tests ─────────────────────────────────────────────────

#[test]
fn test_help_shows_new_env_vars() {
    Command::cargo_bin("llama")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("LLAMA_SYSTEM_PROMPT_FILE"))
        .stdout(predicate::str::contains("LLAMA_DEVICE"))
        .stdout(predicate::str::contains("LLAMA_PROMPT_TEMPLATE_FILE"))
        .stdout(predicate::str::contains("LLAMA_PROMPT_TEMPLATE"))
        .stdout(predicate::str::contains("LLAMA_TEMPERATURE"))
        .stdout(predicate::str::contains("LLAMA_MAX_TOKENS"))
        .stdout(predicate::str::contains("LLAMA_CTX_OVERFLOW"))
        .stdout(predicate::str::contains("LLAMA_STOP"))
        .stdout(predicate::str::contains("LLAMA_TOP_K"))
        .stdout(predicate::str::contains("LLAMA_REPEAT_PENALTY"))
        .stdout(predicate::str::contains("LLAMA_PRESENCE_PENALTY"))
        .stdout(predicate::str::contains("LLAMA_TOP_P"))
        .stdout(predicate::str::contains("LLAMA_MIN_P"))
        .stdout(predicate::str::contains("LLAMA_LOG_VERBOSITY"));
}

#[test]
fn test_ls_shows_models() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("test.gguf"), vec![0u8; 1024]).unwrap();

    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", tmp.path().to_str().unwrap())
        .args(["ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains("test"));
}
