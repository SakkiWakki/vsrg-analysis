//! PyO3 extension exposing a direct osu! memory reader.
//!
//! The plugin calls into this crate to avoid the tosu HTTP hop. Every
//! public PyO3 function returns either a plain value or a Python-side
//! exception — we keep the API small because each addition means one
//! more thing that can drift against osu! binary updates.
//!
//! v1 surface (Linux/wine only, mania-only):
//!   - find_osu_pid() -> Optional[int]
//!   - resolve(pid) -> ResolvedHandle      # cached signature addresses
//!   - read_state(handle) -> dict           # live mania gameplay state
//!   - scan_for_pattern(pid, pattern_hex)   # debug/verify
//!   - read_u32(pid, addr)                  # debug/verify

#[cfg(not(target_os = "linux"))]
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod pattern;
mod process;
mod signatures;

#[cfg(target_os = "linux")]
mod linux_mem;

#[cfg(target_os = "linux")]
mod reader;

/// Locate a running osu! process. Returns None if not found, or the
/// first matching PID if multiple exist (pathological — osu! usually
/// singletons via wine).
#[pyfunction]
fn find_osu_pid() -> PyResult<Option<u32>> {
    #[cfg(target_os = "linux")]
    {
        Ok(process::find_osu_pid_linux())
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(PyNotImplementedError::new_err(
            "find_osu_pid: only Linux/wine is implemented in v1",
        ))
    }
}

/// Scan the tosu-compatible Linux memory regions of ``pid`` for
/// ``pattern_hex``.
///
/// Pattern syntax: whitespace-separated hex bytes, with ``??`` as a
/// wildcard byte. Example: ``"F8 01 74 04 ?? 65 8B"``. Returns the
/// absolute address of the match, or None if not found.
///
/// Debug/verification tool only — real callers should use ``resolve``
/// and let ``signatures.rs`` own the byte literals.
#[pyfunction]
fn scan_for_pattern(pid: u32, pattern_hex: &str) -> PyResult<Option<u64>> {
    let pat = pattern::parse(pattern_hex).map_err(|e| PyValueError::new_err(e.to_string()))?;
    #[cfg(target_os = "linux")]
    {
        linux_mem::scan(pid, &pat).map_err(|e| PyOSError::new_err(e.to_string()))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pid, pat);
        Err(PyNotImplementedError::new_err(
            "scan_for_pattern: only Linux/wine is implemented in v1",
        ))
    }
}

/// Debug-only: return every match of ``pattern_hex`` in the
/// tosu-compatible scan regions. Useful for diagnosing signature
/// collisions (multiple hits means the pattern is too short or needs
/// an anchor).
#[pyfunction]
fn scan_all_for_pattern(pid: u32, pattern_hex: &str) -> PyResult<Vec<u64>> {
    let pat = pattern::parse(pattern_hex).map_err(|e| PyValueError::new_err(e.to_string()))?;
    #[cfg(target_os = "linux")]
    {
        linux_mem::scan_all(pid, &pat).map_err(|e| PyOSError::new_err(e.to_string()))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pid, pat);
        Err(PyNotImplementedError::new_err(
            "scan_all_for_pattern: only Linux/wine is implemented in v1",
        ))
    }
}

/// Read a little-endian u32 at ``addr`` in ``pid``'s address space.
/// Returns None if the read fails.
#[pyfunction]
fn read_u32(pid: u32, addr: u64) -> PyResult<Option<u32>> {
    #[cfg(target_os = "linux")]
    {
        let mut buf = [0u8; 4];
        match linux_mem::read_exact(pid, addr, &mut buf) {
            Ok(()) => Ok(Some(u32::from_le_bytes(buf))),
            Err(_) => Ok(None),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (pid, addr);
        Err(PyNotImplementedError::new_err(
            "read_u32: only Linux/wine is implemented in v1",
        ))
    }
}

/// Cached signature resolutions. Call ``resolve(pid)`` once per
/// osu!-lifetime; pass the handle to ``read_state`` each tick. A new
/// resolve is required after the player restarts osu! (new PID, new
/// addresses).
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ResolvedHandle {
    #[cfg(target_os = "linux")]
    inner: reader::Resolved,
}

#[pymethods]
impl ResolvedHandle {
    #[getter]
    fn pid(&self) -> u32 {
        #[cfg(target_os = "linux")]
        {
            self.inner.pid
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }
    #[getter]
    fn rulesets_ptr(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            self.inner.rulesets_ptr
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }
    #[getter]
    fn base_addr(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            self.inner.base_addr
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }
    #[getter]
    fn status_ptr(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            self.inner.status_ptr
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }
}

/// Scan ``pid`` for every signature we need and return a handle. Raises
/// ``OSError`` on syscall failure or if a required pattern is missing
/// (which means osu! updated and ``signatures.rs`` needs a refresh).
#[pyfunction]
fn resolve(pid: u32) -> PyResult<ResolvedHandle> {
    #[cfg(target_os = "linux")]
    {
        let inner = reader::resolve(pid).map_err(|e| PyOSError::new_err(e.to_string()))?;
        Ok(ResolvedHandle { inner })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        Err(PyNotImplementedError::new_err(
            "resolve: only Linux/wine is implemented in v1",
        ))
    }
}

/// Read the current mania gameplay state via a resolved handle.
///
/// Returns a dict with the keys consumed by ``LiveSnapshot``:
///
///   - ``playing``: bool — False means the pointer chain is null
///     (player on menu / between maps).
///   - ``combo``, ``max_combo``: int
///   - ``mode``: int — osu! ruleset id (0=std, 1=taiko, 2=catch, 3=mania)
///   - ``accuracy``: float (percent, 0..100)
///   - ``hit_300``/``hit_100``/``hit_50``/``hit_miss``/``hit_geki``/``hit_katu``: int
///   - ``hit_errors_ms``: list[int] — per-hit timing offsets in ms
#[pyfunction]
fn read_state<'py>(py: Python<'py>, handle: &ResolvedHandle) -> PyResult<Bound<'py, PyDict>> {
    #[cfg(target_os = "linux")]
    {
        let s = reader::read_mania_state(&handle.inner)
            .map_err(|e| PyOSError::new_err(e.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("playing", s.playing)?;
        d.set_item("combo", s.combo)?;
        d.set_item("max_combo", s.max_combo)?;
        d.set_item("mode", s.mode)?;
        d.set_item("accuracy", s.accuracy)?;
        d.set_item("hit_300", s.hit_300)?;
        d.set_item("hit_100", s.hit_100)?;
        d.set_item("hit_50", s.hit_50)?;
        d.set_item("hit_geki", s.hit_geki)?;
        d.set_item("hit_katu", s.hit_katu)?;
        d.set_item("hit_miss", s.hit_miss)?;
        d.set_item("hit_errors_ms", s.hit_errors_ms)?;
        d.set_item("map_md5", s.map_md5)?;
        d.set_item("map_title", s.map_title)?;
        d.set_item("map_cs", s.map_cs)?;
        d.set_item("game_state", s.game_state)?;
        d.set_item("in_gameplay", s.in_gameplay)?;
        Ok(d)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (py, handle);
        Err(PyNotImplementedError::new_err(
            "read_state: only Linux/wine is implemented in v1",
        ))
    }
}

#[pymodule]
fn osu_memory_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_osu_pid, m)?)?;
    m.add_function(wrap_pyfunction!(scan_for_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(scan_all_for_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(read_u32, m)?)?;
    m.add_function(wrap_pyfunction!(resolve, m)?)?;
    m.add_function(wrap_pyfunction!(read_state, m)?)?;
    m.add_class::<ResolvedHandle>()?;
    Ok(())
}
