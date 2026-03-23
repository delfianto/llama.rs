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
        .stdout(predicate::str::contains("interactive REPL"));
}

#[test]
fn test_serve_help() {
    Command::cargo_bin("llama")
        .unwrap()
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("API server"));
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
    let org_dir = tmp.path().join("org");
    std::fs::create_dir_all(&org_dir).unwrap();
    std::fs::write(org_dir.join("repo-Q4_K_M.gguf"), vec![0u8; 2048]).unwrap();

    Command::cargo_bin("llama")
        .unwrap()
        .env("LLAMA_MODELS_DIR", tmp.path().to_str().unwrap())
        .args(["ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains("org/repo-Q4_K_M"));
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
        .stdout(predicate::str::contains("NAME"))
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
