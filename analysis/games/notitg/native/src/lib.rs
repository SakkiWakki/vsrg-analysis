//! PyO3 extension: the native script->timeline compiler CORE.
//!
//! Step 1 surface (pure-logic, no live engine): a `NativeInterpreter` that seeds
//! globals from a Python dict, marshals a parsed Python AST body ONCE, runs it
//! natively (own scope/tables/builtins), and reads globals back - so the Python
//! parity harness can diff it against `frame_eval` on pure-logic bodies. The
//! live-engine frontier (QUERY_LIVE/poke/...) is step 2: the `Frontier` trait is
//! already the seam; a Python implementor drops in without changing the core.

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod ast;
mod builtins;
mod eval;
mod frontier;
mod marshal;
mod pyconv;
mod scope;
mod table;
mod value;

use crate::eval::Interp;
use crate::frontier::NoFrontier;
use crate::scope::{GlobalStore, MapStore};

/// A persistent native interpreter over a private global store (pure-logic).
/// One instance runs a body repeatedly (accumulator globals carry across calls),
/// mirroring the Python `CompiledBody`'s persistent-interpreter model.
///
/// `unsendable`: the value model is `Rc`/`RefCell` (single-threaded, like the
/// Python interpreter under the GIL). PyO3 enforces this - the interpreter is
/// only ever touched from the thread that created it, which is exactly how the
/// sim drives it.
#[pyclass(unsendable)]
struct NativeInterpreter {
    globals: MapStore,
}

#[pymethods]
impl NativeInterpreter {
    #[new]
    fn new() -> NativeInterpreter {
        NativeInterpreter {
            globals: MapStore::new(),
        }
    }

    /// Seed a global from Python (a driver value like `beat`, an accumulator
    /// seed). Overwrites any prior binding.
    fn set_global(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.globals.set(name, pyconv::value_from_py(value)?);
        Ok(())
    }

    /// Read a global back (for the parity harness). `unresolved` is
    /// frame_eval's UNRESOLVED sentinel, returned as-is for an unbound name so
    /// identity comparison matches the Python oracle.
    fn get_global<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        unresolved: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyconv::value_to_py(py, &self.globals.get(name), unresolved)
    }

    /// Marshal `py_body` (a list of parsed AST statement nodes) and run it once
    /// against the persistent globals. No live engine - a body touching only
    /// clock/math/const/own-tables runs fully native; anything reaching the
    /// frontier (a method read/poke) is inert here (UNRESOLVED/no-op).
    fn run_body(&mut self, py_body: &Bound<'_, PyAny>) -> PyResult<()> {
        let body = marshal::marshal_body(py_body)?;
        let mut frontier = NoFrontier;
        let mut interp = Interp::new(&mut self.globals, &mut frontier);
        interp.run(&body);
        Ok(())
    }

    /// Run `py_body` and return ALL globals it wrote as a Python dict (name ->
    /// value), the parity harness's primary readout.
    fn run_body_globals<'py>(
        &mut self,
        py: Python<'py>,
        py_body: &Bound<'py, PyAny>,
        unresolved: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.run_body(py_body)?;
        let out = PyDict::new(py);
        for (name, value) in self.globals.iter() {
            out.set_item(name, pyconv::value_to_py(py, value, unresolved)?)?;
        }
        Ok(out)
    }
}

/// Expose the crate's name so Python can confirm the native module loaded.
#[pyfunction]
fn backend_name() -> &'static str {
    "notitg_frame_native"
}

#[pymodule]
fn notitg_frame_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeInterpreter>()?;
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    Ok(())
}
