use indicatif::{ProgressBar, ProgressStyle};

/// Create a progress bar for a download with known total size.
pub fn create_download_bar(total_bytes: u64) -> ProgressBar {
    let bar = ProgressBar::new(total_bytes);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .expect("valid template")
            .progress_chars("█▓░"),
    );
    bar
}

/// Create a spinner for a download with unknown total size.
pub fn create_download_spinner() -> ProgressBar {
    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec})")
            .expect("valid template"),
    );
    bar
}
