//! The live frontier, implemented over a Python bridge object. This is the
//! step-2 abstraction layer: the Rust core calls the `Frontier` trait; a
//! `PyFrontier` marshals each call to a Python object (the bridge) that owns the
//! live `NotitgGuardSurface` + a handle registry, then marshals the result back.
//!
//! HANDLES: the core represents any Python object the surface returns/consumes
//! (an actor recorder, a lupa table, a lupa function) as an opaque
//! `Value::Handle(id)` / `Callable::Host(id)`. The bridge keeps `{id: obj}` and
//! resolves ids at the boundary, so the core never holds a Python reference -
//! the frontier is the only place Python and Rust values meet.
//!
//! The bridge's Python protocol (methods the object must expose):
//!   symbol(name)            -> value
//!   call(name, args)        -> value
//!   method(recv, name, args)-> value      # recv/args carry handle ids
//!   poke(recv, name, args)  -> None
//!   index(handle, key)      -> value
//!   set_index(h, key, val)  -> bool
//!   iter_table(handle)      -> list[list[value]] | None
//!   call_host(id, args)     -> value
//! where every `value` at the boundary is either a primitive (num/str/bool/None)
//! or a tagged handle dict `{"__handle__": id}` (a Python object the core sees
//! as opaque). The marshalling is symmetric in both directions.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::frontier::Frontier;
use crate::pyconv;
use crate::value::Value;

pub struct PyFrontier {
    bridge: Py<PyAny>,
    /// The shared UNRESOLVED sentinel (frame_eval's), so the boundary marshals
    /// UNRESOLVED to the exact object the Python side compares by identity.
    unresolved: Py<PyAny>,
}

impl PyFrontier {
    pub fn new(bridge: Py<PyAny>, unresolved: Py<PyAny>) -> PyFrontier {
        PyFrontier { bridge, unresolved }
    }

    /// Call `method_name(*args)` on the bridge and marshal the result back to a
    /// core Value. Any Python/marshalling error collapses to UNRESOLVED (the
    /// "skip, do not guess" floor - a frontier error never crashes the tick).
    fn call_bridge(&self, method_name: &str, args_builder: impl FnOnce(Python) -> PyResult<Py<PyTuple>>) -> Value {
        Python::attach(|py| {
            let result = (|| -> PyResult<Value> {
                let args = args_builder(py)?;
                let ret = self.bridge.bind(py).call_method1(method_name, args.bind(py))?;
                pyconv::value_from_frontier(&ret, self.unresolved.bind(py))
            })();
            result.unwrap_or(Value::Unresolved)
        })
    }
}

impl Frontier for PyFrontier {
    fn symbol(&mut self, name: &str) -> Value {
        self.call_bridge("symbol", |py| Ok(PyTuple::new(py, [name])?.unbind()))
    }

    fn backs_globals(&self) -> bool {
        true
    }

    fn global_get(&mut self, name: &str) -> Value {
        self.call_bridge("global_get", |py| Ok(PyTuple::new(py, [name])?.unbind()))
    }

    fn global_set(&mut self, name: &str, value: &Value) {
        let _ = self.call_bridge("global_set", |py| {
            let val = pyconv::value_to_frontier(py, value, self.unresolved.bind(py))?;
            Ok(PyTuple::new(py, [name.into_pyobject(py)?.into_any(), val])?.unbind())
        });
    }

    fn call(&mut self, name: &str, args: &[Value]) -> Value {
        self.call_bridge("call", |py| {
            let arglist = pyconv::values_to_frontier(py, args, self.unresolved.bind(py))?;
            Ok(PyTuple::new(py, [name.into_pyobject(py)?.into_any(), arglist.into_any()])?.unbind())
        })
    }

    fn method(&mut self, recv: &Value, name: &str, args: &[Value]) -> Value {
        self.call_bridge("method", |py| {
            let recv_py = pyconv::value_to_frontier(py, recv, self.unresolved.bind(py))?;
            let arglist = pyconv::values_to_frontier(py, args, self.unresolved.bind(py))?;
            Ok(PyTuple::new(
                py,
                [recv_py, name.into_pyobject(py)?.into_any(), arglist.into_any()],
            )?
            .unbind())
        })
    }

    fn poke(&mut self, recv: &Value, name: &str, args: &[Value]) {
        let _ = self.call_bridge("poke", |py| {
            let recv_py = pyconv::value_to_frontier(py, recv, self.unresolved.bind(py))?;
            let arglist = pyconv::values_to_frontier(py, args, self.unresolved.bind(py))?;
            Ok(PyTuple::new(
                py,
                [recv_py, name.into_pyobject(py)?.into_any(), arglist.into_any()],
            )?
            .unbind())
        });
    }

    fn index(&mut self, handle: u64, key: &Value) -> Value {
        self.call_bridge("index", |py| {
            let key_py = pyconv::value_to_frontier(py, key, self.unresolved.bind(py))?;
            Ok(PyTuple::new(py, [handle.into_pyobject(py)?.into_any(), key_py])?.unbind())
        })
    }

    fn set_index(&mut self, handle: u64, key: &Value, value: &Value) -> bool {
        let result = Python::attach(|py| {
            let key_py = pyconv::value_to_frontier(py, key, self.unresolved.bind(py))?;
            let val_py = pyconv::value_to_frontier(py, value, self.unresolved.bind(py))?;
            let args = PyTuple::new(py, [handle.into_pyobject(py)?.into_any(), key_py, val_py])?;
            let ret = self.bridge.bind(py).call_method1("set_index", args)?;
            ret.extract::<bool>()
        });
        result.unwrap_or(false)
    }

    fn iter_table(&mut self, handle: u64) -> Option<Vec<Vec<Value>>> {
        Python::attach(|py| {
            let ret = self
                .bridge
                .bind(py)
                .call_method1("iter_table", (handle,))
                .ok()?;
            if ret.is_none() {
                return None;
            }
            let list = ret.cast::<PyList>().ok()?;
            let mut rows = Vec::new();
            for row in list.iter() {
                let row_list = row.cast::<PyList>().ok()?;
                let mut kv = Vec::new();
                for item in row_list.iter() {
                    kv.push(
                        pyconv::value_from_frontier(&item, self.unresolved.bind(py))
                            .unwrap_or(Value::Unresolved),
                    );
                }
                rows.push(kv);
            }
            Some(rows)
        })
    }

    fn call_host(&mut self, id: u64, args: &[Value]) -> Value {
        self.call_bridge("call_host", |py| {
            let arglist = pyconv::values_to_frontier(py, args, self.unresolved.bind(py))?;
            Ok(PyTuple::new(py, [id.into_pyobject(py)?.into_any(), arglist.into_any()])?.unbind())
        })
    }

    fn snapshot_global(&mut self, name: &str) -> Value {
        // The bridge returns a plain Python list/dict tree (nested), which
        // `value_from_py` deep-marshals into native LuaTables - so the snapshot
        // is a fully native structure the core owns.
        Python::attach(|py| {
            let result = (|| -> PyResult<Value> {
                let ret = self.bridge.bind(py).call_method1("snapshot_global", (name,))?;
                if ret.is(self.unresolved.bind(py)) || ret.is_none() {
                    return Ok(Value::Unresolved);
                }
                pyconv::value_from_py(&ret)
            })();
            result.unwrap_or(Value::Unresolved)
        })
    }
}
