#![allow(dead_code)] // Stubs — remove as modules get implemented
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::doc_markdown,
    clippy::cast_possible_truncation
)]

pub mod api;
pub mod cli;
pub mod config;
pub mod download;
pub mod error;
pub mod model;
pub mod process;
