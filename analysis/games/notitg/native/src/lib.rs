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

mod analyze;
mod ast;
mod builtins;
mod compile;
mod eval;
mod frontier;
mod marshal;
mod pyconv;
mod pyfrontier;
mod scope;
mod table;
mod value;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::ast::Stmt;
use crate::eval::Interp;
use crate::frontier::NoFrontier;
use crate::pyfrontier::PyFrontier;
use crate::scope::{GlobalStore, MapStore};
use crate::value::Value;

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
    /// The body compiled ONCE (marshalled AST + its snapshottable-name set), so
    /// the per-tick path re-uses it - no re-marshal, no re-analysis per tick.
    compiled: Option<Rc<[Stmt]>>,
    /// The body's AST closure-compiled ONCE (compile.rs) - the per-tick run is a
    /// direct call chain, no per-node match/AST-walk.
    compiled_fn: Option<compile::CStmt>,
    snapshottable: HashSet<String>,
    /// name -> snapshotted native data table, persisted across ticks.
    snapshots: HashMap<String, Value>,
    /// Per-tick DRIVER cache (mod_time/beat/...): seeded by the sim each tick,
    /// read native so the hot driver reads (mod_time was 134 crossings/tick on
    /// gat) do not cross to the host env.
    tick_cache: HashMap<String, Value>,
    /// Per-tick ACTOR-value cache (handle, verb) -> current value, seeded by the
    /// sim after drain for the learned read set, so GetX/GetY/... stay native.
    actor_cache: HashMap<(u64, String), Value>,
    /// The (handle, verb) actor reads the last run made - the sim drains this to
    /// learn which values to seed next tick.
    actor_reads: std::cell::RefCell<Vec<(u64, String)>>,
}

#[pymethods]
impl NativeInterpreter {
    #[new]
    fn new() -> NativeInterpreter {
        NativeInterpreter {
            globals: MapStore::new(),
            compiled: None,
            compiled_fn: None,
            snapshottable: HashSet::new(),
            snapshots: HashMap::new(),
            tick_cache: HashMap::new(),
            actor_cache: HashMap::new(),
            actor_reads: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Seed a per-tick DRIVER value (mod_time/beat) read natively this tick,
    /// bypassing the frontier crossing. Only for values the sim owns and the
    /// body never writes (the driver clocks); call once per tick before running.
    fn set_tick_driver(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.tick_cache
            .insert(name.to_string(), pyconv::value_from_py(value)?);
        Ok(())
    }

    /// Seed an actor's current value for `(handle, verb)` this tick, read native
    /// by a GetX/GetY/... on that handle. The sim calls this after drain for the
    /// learned read set (from `take_actor_reads`).
    fn seed_actor_value(
        &mut self,
        handle: u64,
        verb: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.actor_cache
            .insert((handle, verb.to_string()), pyconv::value_from_py(value)?);
        Ok(())
    }

    /// Clear the actor-value cache (call before re-seeding each tick, so a stale
    /// value from a no-longer-read actor cannot linger).
    fn clear_actor_cache(&mut self) {
        self.actor_cache.clear();
    }

    /// Drain and return the (handle, verb) actor reads the last run made, as a
    /// list of (handle, verb) tuples - the sim uses them to learn which values
    /// to seed next tick.
    fn take_actor_reads(&self) -> Vec<(u64, String)> {
        std::mem::take(&mut self.actor_reads.borrow_mut())
    }

    /// Compile a body ONCE: marshal the Python AST and compute the snapshottable
    /// data-table names (referenced, never written). Subsequent ticks call
    /// `run_compiled_frontier`, re-using this without re-marshalling.
    fn compile_body(&mut self, py_body: &Bound<'_, PyAny>) -> PyResult<()> {
        let body = marshal::marshal_body(py_body)?;
        self.snapshottable = analyze::snapshottable_names(&body);
        self.snapshots.clear();
        self.compiled_fn = Some(compile::compile_block(&body));
        self.compiled = Some(body);
        Ok(())
    }

    /// Run the compiled body against the live frontier (the per-tick hot path).
    /// Snapshottable data tables are copied native on first sight and cached, so
    /// their `v[i][j]` reads never cross the frontier again.
    fn run_compiled_frontier(
        &mut self,
        bridge: &Bound<'_, PyAny>,
        unresolved: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let body_fn = match &self.compiled_fn {
            Some(f) => f.clone(),
            None => return Ok(()),
        };
        self.actor_reads.borrow_mut().clear();
        let mut frontier = PyFrontier::new(bridge.clone().unbind(), unresolved.clone().unbind());
        let mut interp = Interp::new(
            &mut self.globals,
            &mut frontier,
            &self.snapshottable,
            &mut self.snapshots,
            &self.tick_cache,
            &self.actor_cache,
            &self.actor_reads,
        );
        interp.run_compiled(&body_fn);
        Ok(())
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
        let empty = HashSet::new();
        let empty_cache = HashMap::new();
        let empty_actor = HashMap::new();
        let empty_reads = std::cell::RefCell::new(Vec::new());
        let mut snaps = HashMap::new();
        let mut interp = Interp::new(&mut self.globals, &mut frontier, &empty, &mut snaps, &empty_cache, &empty_actor, &empty_reads);
        interp.run(&body);
        Ok(())
    }

    /// Run a marshalled body against the LIVE frontier: `bridge` is a Python
    /// object implementing the frontier protocol (symbol/call/method/poke/
    /// index/set_index/iter_table/call_host over the live NotitgGuardSurface),
    /// `unresolved` is frame_eval's sentinel. The core owns scope/tables/math;
    /// every live-engine crossing routes to `bridge`. This is the step-2 seam -
    /// the Rust residue tick loop driving the real sim.
    fn run_body_frontier(
        &mut self,
        py_body: &Bound<'_, PyAny>,
        bridge: &Bound<'_, PyAny>,
        unresolved: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let body = marshal::marshal_body(py_body)?;
        let mut frontier = PyFrontier::new(bridge.clone().unbind(), unresolved.clone().unbind());
        let empty = HashSet::new();
        let empty_cache = HashMap::new();
        let empty_actor = HashMap::new();
        let empty_reads = std::cell::RefCell::new(Vec::new());
        let mut snaps = HashMap::new();
        let mut interp = Interp::new(&mut self.globals, &mut frontier, &empty, &mut snaps, &empty_cache, &empty_actor, &empty_reads);
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
