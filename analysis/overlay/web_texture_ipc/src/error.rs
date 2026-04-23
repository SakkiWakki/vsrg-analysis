//! Error type for the crate. Pyo3-friendly: every variant maps to a
//! meaningful Python-side message via ``From<Error> for PyErr``.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("socket I/O failed: {0}")]
    SocketIo(#[from] std::io::Error),

    #[error("EGL error: {0}")]
    Egl(String),

    #[error(
        "required EGL extension missing: {0}. \
         This driver probably doesn't support dmabuf export; \
         the caller should fall back to a CPU-readback backend."
    )]
    MissingEglExtension(&'static str),

    #[error("dmabuf export returned no plane; driver bug or unsupported format")]
    EmptyExport,

    #[error("invalid argument: {0}")]
    InvalidArg(&'static str),

    #[error("channel closed")]
    Closed,
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for pyo3::PyErr {
    fn from(value: Error) -> Self {
        use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
        match value {
            Error::SocketIo(e) => PyOSError::new_err(e.to_string()),
            Error::InvalidArg(s) => PyValueError::new_err(s),
            Error::Closed => PyRuntimeError::new_err("channel already closed"),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}
