//! The MIGRATION FRONTIER - the trait the core calls back into for everything it
//! does NOT own: live-engine reads (`method`), effects (`poke`), host-table
//! index/write/iterate, host-function calls, and unresolved symbol/free-call
//! resolution. Drawn wherever Python currently stops and Rust begins; today the
//! implementor is Python (the Surface/SimActor), so these cross out to Python.
//! As each piece ports, the SAME trait gets a Rust implementor and the crossing
//! goes native - the core never changes (COMPILER_CONTRACT).
//!
//! `NoFrontier` is the step-1 stub: every crossing yields UNRESOLVED / no-op, so
//! a PURE-LOGIC body (no live engine) runs fully native and can be parity-tested
//! against `frame_eval` before the live Python frontier is wired (step 2).

use crate::value::Value;

pub trait Frontier {
    /// A bare symbol not bound in scope/globals: a driver clock (`beat`), an
    /// actor global, or a host function. UNRESOLVED when the frontier does not
    /// know it (skip, do not guess).
    fn symbol(&mut self, name: &str) -> Value;

    /// Whether the frontier BACKS globals (a host env the load pass populated).
    /// When true, the core routes global reads/writes to `global_get`/
    /// `global_set` so an accumulator the body writes persists to the SAME
    /// namespace the guards + load globals share. The NoFrontier stub returns
    /// false (the core's own MapStore holds globals - the pure-logic path).
    fn backs_globals(&self) -> bool {
        false
    }

    /// Read a global from the host env (frontier-backed mode). UNRESOLVED for an
    /// absent name.
    fn global_get(&mut self, _name: &str) -> Value {
        Value::Unresolved
    }

    /// Write a global to the host env (frontier-backed mode).
    fn global_set(&mut self, _name: &str, _value: &Value) {}

    /// A free call `name(args)` the core did not service (`perframe`), else
    /// UNRESOLVED.
    fn call(&mut self, name: &str, args: &[Value]) -> Value;

    /// A method in VALUE position `recv:name(args)` - a live getter read
    /// (`self:GetX()`). UNRESOLVED for a nil/absent read.
    fn method(&mut self, recv: &Value, name: &str, args: &[Value]) -> Value;

    /// A method in EFFECT position `recv:name(args)` - a poke (a setter, a
    /// queuecommand). No return; an unresolved recv is dropped.
    fn poke(&mut self, recv: &Value, name: &str, args: &[Value]);

    /// `host_handle[key]` read. UNRESOLVED for absent (the core maps that to nil
    /// when the base was resolved).
    fn index(&mut self, handle: u64, key: &Value) -> Value;

    /// `host_handle[key] = value`. Returns whether the write landed.
    fn set_index(&mut self, handle: u64, key: &Value, value: &Value) -> bool;

    /// (k, v) rows for a generic-for over a host table, or None when it is not
    /// a host table the frontier owns.
    fn iter_table(&mut self, handle: u64) -> Option<Vec<Vec<Value>>>;

    /// Call a host function by its opaque id (a lupa closure). The frontier
    /// marshals UNRESOLVED->nil so a host fn never sees the sentinel.
    fn call_host(&mut self, id: u64, args: &[Value]) -> Value;

    /// Deep-snapshot a load-populated host DATA TABLE named `name` into a native
    /// value (recursively: nested tables become native tables, leaves are
    /// primitives; a host function inside stays a `Func(Host)`). Returns
    /// UNRESOLVED when `name` is not a host table. The core caches the result so
    /// `v[i][j]` reads are then FULLY NATIVE - no per-element frontier crossing
    /// (the measured 15K index crossings/tick). Only sound for tables the body
    /// does not write (the caller gates on a static "never assigned" check).
    fn snapshot_global(&mut self, _name: &str) -> Value {
        Value::Unresolved
    }

    /// Whether a frontier call RAISED (a host function erroring, e.g. a Lua
    /// `attempt to compare number with nil`). The Python interpreter lets such
    /// an error abort the WHOLE body tick (it propagates to the fault handler),
    /// so the core checks this after a frontier call and stops the run to match
    /// - a swallowed error would let the body compute values Python never
    /// reaches. NoFrontier never faults.
    fn aborted(&self) -> bool {
        false
    }
}

/// The pure-logic stub: no live engine. Every crossing is inert, so a body that
/// touches only clock+math+const+own-tables runs entirely in the core.
pub struct NoFrontier;

impl Frontier for NoFrontier {
    fn symbol(&mut self, _name: &str) -> Value {
        Value::Unresolved
    }
    fn call(&mut self, _name: &str, _args: &[Value]) -> Value {
        Value::Unresolved
    }
    fn method(&mut self, _recv: &Value, _name: &str, _args: &[Value]) -> Value {
        Value::Unresolved
    }
    fn poke(&mut self, _recv: &Value, _name: &str, _args: &[Value]) {}
    fn index(&mut self, _handle: u64, _key: &Value) -> Value {
        Value::Unresolved
    }
    fn set_index(&mut self, _handle: u64, _key: &Value, _value: &Value) -> bool {
        false
    }
    fn iter_table(&mut self, _handle: u64) -> Option<Vec<Vec<Value>>> {
        None
    }
    fn call_host(&mut self, _id: u64, _args: &[Value]) -> Value {
        Value::Unresolved
    }
}
