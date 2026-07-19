//! Value <-> Python conversion at the module boundary. Numbers/strings/bools/
//! nil marshal directly; a core `Table` becomes a Python dict-ish (list for a
//! pure array, else a dict) for inspection; UNRESOLVED marshals to a shared
//! sentinel so the Python parity harness can compare against `frame_eval`'s
//! UNRESOLVED. Handles/funcs are opaque on the way OUT (a parity test on
//! pure-logic bodies never produces them).

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyString};

use crate::table::Key;
use crate::value::{Callable, Value};

/// Python -> core Value (seeding globals). None/True/False/int/float/str map
/// directly; a list becomes an array table; anything else -> UNRESOLVED (the
/// pure-logic path does not seed live handles).
pub fn value_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Nil);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        // int and float both land here; bool was handled above.
        return Ok(Value::Num(f));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::str(s.to_string_lossy().into_owned()));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut t = crate::table::LuaTable::new();
        for item in list.iter() {
            t.append(value_from_py(&item)?);
        }
        return Ok(Value::Table(std::rc::Rc::new(std::cell::RefCell::new(t))));
    }
    Ok(Value::Unresolved)
}

/// core Value -> Python, for reading globals back in the parity harness. A
/// number is a float, a string a str, nil is None; UNRESOLVED is the sentinel
/// object passed in from Python (so identity compares equal to frame_eval's).
pub fn value_to_py<'py>(
    py: Python<'py>,
    v: &Value,
    unresolved: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let obj = match v {
        Value::Num(n) => PyFloat::new(py, *n).into_any(),
        Value::Str(s) => PyString::new(py, s).into_any(),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Nil => py.None().into_bound(py),
        Value::Unresolved => unresolved.clone(),
        Value::Table(t) => table_to_py(py, &t.borrow(), unresolved)?,
        Value::Func(_) | Value::Handle(_) => py.None().into_bound(py),
    };
    Ok(obj)
}

/// A core table -> a Python list when it is a pure 1..n array, else a dict.
/// Enough for the parity harness to inspect a body's table state.
fn table_to_py<'py>(
    py: Python<'py>,
    t: &crate::table::LuaTable,
    unresolved: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = t.length();
    let pairs = t.pairs();
    let is_pure_array = pairs.len() as i64 == n
        && pairs
            .iter()
            .all(|(k, _)| matches!(k, Value::Num(x) if x.fract() == 0.0));
    if is_pure_array {
        let list = PyList::empty(py);
        for i in 1..=n {
            list.append(value_to_py(py, &t.get(&Key::Int(i)), unresolved)?)?;
        }
        Ok(list.into_any())
    } else {
        let dict = PyDict::new(py);
        for (k, v) in pairs {
            let key = value_to_py(py, &k, unresolved)?;
            let val = value_to_py(py, &v, unresolved)?;
            dict.set_item(key, val)?;
        }
        Ok(dict.into_any())
    }
}

// -- frontier marshalling (step 2) -------------------------------------------
//
// At the live frontier, a value crossing to/from Python is either a PRIMITIVE
// (num/str/bool/None/UNRESOLVED) or an opaque HANDLE (a Python object the core
// cannot represent - an actor recorder, a lupa table/function). The bridge tags
// a handle as a Python object carrying `__handle__` (a host object/table id) or
// `__host_fn__` (a host callable id); the core marshals those to `Value::Handle`
// / `Callable::Host`. Everything else marshals as a plain value.

/// core Value -> a Python arg for the bridge. A handle becomes a tagged dict
/// `{"__handle__": id}` / `{"__host_fn__": id}` the bridge resolves to its
/// object; a table crosses as its list/dict form (rare at the frontier).
pub fn value_to_frontier<'py>(
    py: Python<'py>,
    v: &Value,
    unresolved: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let obj = match v {
        Value::Handle(id) => tagged(py, "__handle__", *id)?,
        Value::Func(Callable::Host(id)) => tagged(py, "__host_fn__", *id)?,
        _ => return value_to_py(py, v, unresolved),
    };
    Ok(obj)
}

fn tagged<'py>(py: Python<'py>, tag: &str, id: u64) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(tag, id)?;
    Ok(dict.into_any())
}

/// A slice of core Values -> a Python list for the bridge.
pub fn values_to_frontier<'py>(
    py: Python<'py>,
    vs: &[Value],
    unresolved: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for v in vs {
        list.append(value_to_frontier(py, v, unresolved)?)?;
    }
    Ok(list)
}

/// A Python value FROM the bridge -> core Value. A tagged handle dict becomes a
/// `Handle`/`Host`; UNRESOLVED (by identity) stays UNRESOLVED; primitives map
/// directly. A dict/list that is NOT a handle tag is a data table (rare) and
/// marshals through `value_from_py`.
pub fn value_from_frontier(obj: &Bound<'_, PyAny>, unresolved: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is(unresolved) {
        return Ok(Value::Unresolved);
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        if let Some(id) = dict.get_item("__handle__")?.and_then(|v| v.extract::<u64>().ok()) {
            return Ok(Value::Handle(id));
        }
        if let Some(id) = dict.get_item("__host_fn__")?.and_then(|v| v.extract::<u64>().ok()) {
            return Ok(Value::Func(Callable::Host(id)));
        }
    }
    value_from_py(obj)
}
