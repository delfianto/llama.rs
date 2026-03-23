pub mod cli;
pub mod health;
pub mod server;

/// Handle to a spawned child process (`std::process` — for terminal-inheriting processes).
pub struct StdProcessHandle {
    pub child: std::process::Child,
    pub pid: u32,
}

/// Handle to a spawned child process (tokio — for background server processes).
pub struct AsyncProcessHandle {
    pub child: tokio::process::Child,
    pub pid: u32,
}
