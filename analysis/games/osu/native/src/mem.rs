//! OS-agnostic re-exports over the per-platform memory backends.
//!
//! `reader.rs` and every other caller goes through `mem::read_exact`
//! and `mem::scan` so the pointer-chain / signature-resolution code
//! stays identical on Linux and Windows. Each backend implements the
//! same three entry points (`read_exact`, `scan`, `scan_all`).

#[cfg(target_os = "linux")]
pub use crate::linux_mem::{read_exact, scan, scan_all};

#[cfg(target_os = "windows")]
pub use crate::windows_mem::{read_exact, scan, scan_all};
