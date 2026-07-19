//! Scope + global store - the core-owned namespace the body's locals and
//! accumulators live in. Mirrors Python `frame_eval.Scope`/`GlobalStore`:
//! a chained lexical scope (child shadows parent) rooted at a persistent global
//! store, so a body's accumulator globals carry ACROSS ticks (the persistent
//! Interpreter reuses one root scope) while a block's locals do not leak.
//!
//! Globals may be BACKED by the frontier (a live host env the Python side owns),
//! so a load-populated global and a per-frame accumulator share one namespace.
//! The core reads/writes globals through a `GlobalStore` trait the frontier can
//! implement; the default is a private map (the pure-logic path).

use std::collections::HashMap;

use crate::value::Value;

/// Where globals live. The default `MapStore` is core-owned; the frontier can
/// supply a host-backed store so globals round-trip with the live Lua env.
pub trait GlobalStore {
    fn has(&self, name: &str) -> bool;
    fn get(&self, name: &str) -> Value;
    fn set(&mut self, name: &str, value: Value);
}

/// A private global namespace (the pure-logic path). `get` returns UNRESOLVED
/// for an absent name (the surface would resolve a driver/clock symbol; here
/// nothing does).
#[derive(Default)]
pub struct MapStore {
    d: HashMap<String, Value>,
}

impl MapStore {
    pub fn new() -> MapStore {
        MapStore { d: HashMap::new() }
    }

    /// Every (name, value) global written - the parity harness's readout.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.d.iter()
    }
}

impl GlobalStore for MapStore {
    fn has(&self, name: &str) -> bool {
        self.d.contains_key(name)
    }
    fn get(&self, name: &str) -> Value {
        self.d.get(name).cloned().unwrap_or(Value::Unresolved)
    }
    fn set(&mut self, name: &str, value: Value) {
        self.d.insert(name.to_string(), value);
    }
}

/// A lexical scope: local bindings + a parent link. The ROOT scope's writes go
/// to the global store (a top-level assign is a global); a child scope's
/// `set_local` binds a new local, and `assign` walks up to the nearest binder or
/// falls to the global store.
///
/// Scopes form a tree by index into an arena (`ScopeArena`) so a closure can
/// capture its defining scope by id without Rc cycles, mirroring the Python
/// object-reference chain.
pub struct Scope {
    pub parent: Option<usize>,
    pub locals: HashMap<String, Value>,
}

impl Scope {
    fn new(parent: Option<usize>) -> Scope {
        Scope {
            parent,
            locals: HashMap::new(),
        }
    }
}

/// Arena of scopes. Index 0 is the persistent root. A block pushes a child; the
/// arena is reset between top-level runs EXCEPT the root (globals persist).
pub struct ScopeArena {
    scopes: Vec<Scope>,
}

impl ScopeArena {
    pub fn new() -> ScopeArena {
        ScopeArena {
            scopes: vec![Scope::new(None)],
        }
    }

    pub const ROOT: usize = 0;

    pub fn child(&mut self, parent: usize) -> usize {
        self.scopes.push(Scope::new(Some(parent)));
        self.scopes.len() - 1
    }

    /// Look up `name` in the LOCAL scope chain only. `Some(value)` when bound as
    /// a local somewhere up the chain, else None (the caller then consults
    /// globals - the core's MapStore or the frontier's host env). Splitting
    /// local-vs-global lookup lets the interpreter route globals to whichever
    /// store backs them without the scope knowing.
    pub fn lookup_local(&self, mut scope: usize, name: &str) -> Option<Value> {
        loop {
            let s = &self.scopes[scope];
            if let Some(v) = s.locals.get(name) {
                return Some(v.clone());
            }
            match s.parent {
                Some(p) => scope = p,
                None => return None,
            }
        }
    }

    /// `local name = v` - bind a new local IN THIS scope.
    pub fn set_local(&mut self, scope: usize, name: &str, value: Value) {
        self.scopes[scope].locals.insert(name.to_string(), value);
    }

    /// Rebind the nearest existing LOCAL up the chain to `value`. Returns true
    /// when a local was rebound; false means `name` is not a local anywhere, so
    /// the caller writes it as a global. Matches Python `Scope.assign`'s
    /// local-first rule.
    pub fn assign_local(&mut self, scope: usize, name: &str, value: &Value) -> bool {
        let mut cur = scope;
        loop {
            if self.scopes[cur].locals.contains_key(name) {
                self.scopes[cur].locals.insert(name.to_string(), value.clone());
                return true;
            }
            match self.scopes[cur].parent {
                Some(p) => cur = p,
                None => return false,
            }
        }
    }

    /// Truncate the arena back to `keep` scopes (drop transient block scopes
    /// after a top-level run; keep the root so globals persist).
    pub fn truncate(&mut self, keep: usize) {
        self.scopes.truncate(keep);
    }
}

impl Default for ScopeArena {
    fn default() -> Self {
        Self::new()
    }
}
