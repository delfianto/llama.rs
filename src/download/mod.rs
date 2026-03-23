pub mod hf;
pub mod progress;

use std::path::Path;

use futures::StreamExt;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use crate::error::LlamaError;

/// Download a file from `url` to `dest` with parallel chunk downloads.
///
/// - Sends a HEAD request to determine file size and range support.
/// - If range requests are supported and size > 100MB, splits into `connections` parallel chunks.
/// - Falls back to single-stream download otherwise.
/// - Shows progress via `indicatif`.
/// - Writes to a temp file first, renames on completion.
/// - Cleans up temp files on error.
pub async fn download_file(
    url: &str,
    dest: &Path,
    connections: u8,
    hf_token: Option<&str>,
) -> anyhow::Result<()> {
    let client = build_client(hf_token)?;

    // HEAD request to get size and check range support
    let head_resp = client.head(url).send().await?;

    match head_resp.status().as_u16() {
        404 => anyhow::bail!(LlamaError::DownloadFailed {
            reason: "Model not found on HuggingFace. Check the repo name and filename.".into(),
        }),
        401 | 403 => anyhow::bail!(LlamaError::HfAccessDenied),
        s if s >= 400 => anyhow::bail!(LlamaError::DownloadFailed {
            reason: format!("HTTP {s}"),
        }),
        _ => {}
    }

    let content_length = head_resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    let accepts_ranges = head_resp
        .headers()
        .get("accept-ranges")
        .and_then(|v| v.to_str().ok())
        == Some("bytes");

    let total_size = content_length.unwrap_or(0);
    let use_parallel = accepts_ranges && total_size > 100 * 1024 * 1024 && connections > 1;

    debug!(
        "Download: size={total_size}, ranges={accepts_ranges}, parallel={use_parallel}, connections={connections}"
    );

    let temp_path = dest.with_extension("part");

    let result = if use_parallel {
        download_parallel(&client, url, &temp_path, total_size, connections).await
    } else {
        download_single(&client, url, &temp_path, total_size).await
    };

    match result {
        Ok(()) => {
            tokio::fs::rename(&temp_path, dest).await?;
            Ok(())
        }
        Err(e) => {
            // Clean up temp file on error
            let _ = tokio::fs::remove_file(&temp_path).await;
            Err(e)
        }
    }
}

/// Single-stream download with progress.
async fn download_single(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    total_size: u64,
) -> anyhow::Result<()> {
    info!("Downloading (single stream)...");

    let bar = if total_size > 0 {
        progress::create_download_bar(total_size)
    } else {
        progress::create_download_spinner()
    };

    let resp = client.get(url).send().await?.error_for_status()?;
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(dest).await?;
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        bar.set_position(downloaded);
    }

    file.flush().await?;
    bar.finish_with_message("done");
    Ok(())
}

/// Parallel chunk download with progress.
async fn download_parallel(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    total_size: u64,
    connections: u8,
) -> anyhow::Result<()> {
    info!("Downloading ({connections} parallel connections)...");

    let bar = progress::create_download_bar(total_size);
    let chunk_size = total_size / u64::from(connections);

    // Create the output file with the full size
    let file = tokio::fs::File::create(dest).await?;
    file.set_len(total_size).await?;
    drop(file);

    let mut tasks = tokio::task::JoinSet::new();

    for i in 0..connections {
        let start = u64::from(i) * chunk_size;
        let end = if i == connections - 1 {
            total_size - 1
        } else {
            start + chunk_size - 1
        };

        let client = client.clone();
        let url = url.to_string();
        let dest = dest.to_path_buf();
        let bar = bar.clone();

        tasks.spawn(async move {
            download_chunk_with_retry(&client, &url, &dest, start, end, &bar, 3).await
        });
    }

    while let Some(result) = tasks.join_next().await {
        result??;
    }

    bar.finish_with_message("done");
    Ok(())
}

/// Download a single byte range, writing to the file at the correct offset. Retries on failure.
async fn download_chunk_with_retry(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    start: u64,
    end: u64,
    bar: &indicatif::ProgressBar,
    max_retries: u32,
) -> anyhow::Result<()> {
    let mut attempts = 0;

    loop {
        match download_chunk(client, url, dest, start, end, bar).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempts += 1;
                if attempts >= max_retries {
                    return Err(e);
                }
                warn!("Chunk {start}-{end} failed (attempt {attempts}/{max_retries}): {e}");
                tokio::time::sleep(std::time::Duration::from_secs(u64::from(attempts))).await;
            }
        }
    }
}

async fn download_chunk(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    start: u64,
    end: u64,
    bar: &indicatif::ProgressBar,
) -> anyhow::Result<()> {
    use tokio::io::AsyncSeekExt;

    let resp = client
        .get(url)
        .header("Range", format!("bytes={start}-{end}"))
        .send()
        .await?
        .error_for_status()?;

    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::OpenOptions::new().write(true).open(dest).await?;
    file.seek(std::io::SeekFrom::Start(start)).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        bar.inc(chunk.len() as u64);
    }

    file.flush().await?;
    Ok(())
}

fn build_client(hf_token: Option<&str>) -> anyhow::Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .user_agent("llama-rs/0.1.0")
        .redirect(reqwest::redirect::Policy::limited(10));

    if let Some(token) = hf_token {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))?,
        );
        builder = builder.default_headers(headers);
    }

    Ok(builder.build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use wiremock::matchers::{header, method};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_single_stream_download() {
        let mock = MockServer::start().await;
        let body = b"fake model content here";

        Mock::given(method("HEAD"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-length", body.len().to_string().as_str()),
            )
            .mount(&mock)
            .await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_slice()))
            .mount(&mock)
            .await;

        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("model.gguf");

        download_file(&format!("{}/model.gguf", mock.uri()), &dest, 1, None)
            .await
            .unwrap();

        assert!(dest.exists());
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn test_download_creates_directory_structure() {
        let mock = MockServer::start().await;
        let body = b"data";

        Mock::given(method("HEAD"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-length", body.len().to_string().as_str()),
            )
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_slice()))
            .mount(&mock)
            .await;

        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("org").join("repo").join("model.gguf");
        tokio::fs::create_dir_all(dest.parent().unwrap())
            .await
            .unwrap();

        download_file(&format!("{}/model.gguf", mock.uri()), &dest, 1, None)
            .await
            .unwrap();

        assert!(dest.exists());
    }

    #[tokio::test]
    async fn test_download_404_error() {
        let mock = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&mock)
            .await;

        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("model.gguf");

        let result = download_file(&format!("{}/model.gguf", mock.uri()), &dest, 1, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_download_401_error() {
        let mock = MockServer::start().await;
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock)
            .await;

        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("model.gguf");

        let result = download_file(&format!("{}/model.gguf", mock.uri()), &dest, 1, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("HF_TOKEN"));
    }

    #[tokio::test]
    async fn test_download_with_auth_token() {
        let mock = MockServer::start().await;
        let body = b"gated model";

        Mock::given(method("HEAD"))
            .and(header("authorization", "Bearer test-token-123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-length", body.len().to_string().as_str()),
            )
            .mount(&mock)
            .await;

        Mock::given(method("GET"))
            .and(header("authorization", "Bearer test-token-123"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.as_slice()))
            .mount(&mock)
            .await;

        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("model.gguf");

        download_file(
            &format!("{}/model.gguf", mock.uri()),
            &dest,
            1,
            Some("test-token-123"),
        )
        .await
        .unwrap();

        assert!(dest.exists());
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[tokio::test]
    async fn test_temp_file_cleaned_on_error() {
        let mock = MockServer::start().await;

        // HEAD succeeds but GET fails
        Mock::given(method("HEAD"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-length", "1000"))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock)
            .await;

        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("model.gguf");
        let temp_path = dest.with_extension("part");

        let _result = download_file(&format!("{}/model.gguf", mock.uri()), &dest, 1, None).await;

        assert!(!dest.exists());
        assert!(!temp_path.exists());
    }
}
