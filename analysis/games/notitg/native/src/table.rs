//! The interpreter's own table type - Lua semantics without a host dependency,
//! so a `{...}` constructor round-trips through `table.insert`, `ipairs`, and
//! index reads natively. Mirrors Python `frame_eval.LuaTable`: one store keyed
//! by a normalized key; a stored nil/UNRESOLVED REMOVES the key (so length stays
//! contiguous and an absent key reads back as nil).
//!
//! The corpus showed 0 hash-keyed constructors, so an integer-keyed array run
//! dominates; we still support string keys (field access `t.foo`) via the same
//! map, keeping the model simple and correct over fast.

use std::collections::HashMap;

use crate::value::Value;

/// A normalized table key: Lua has no int/float distinction, so a whole-valued
/// float folds to the integer key (keeps array runs contiguous).
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Int(i64),
    Str(String),
}

impl Key {
    pub fn from_value(v: &Value) -> Option<Key> {
        match v {
            Value::Num(n) => Some(Key::from_num(*n)),
            Value::Str(s) => Some(Key::Str(s.to_string())),
            _ => None,
        }
    }

    pub fn from_num(n: f64) -> Key {
        if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
            Key::Int(n as i64)
        } else {
            // A non-integer numeric key is rare; stringify it for storage.
            Key::Str(crate::value::num_to_lua_string(n))
        }
    }
}

#[derive(Default)]
pub struct LuaTable {
    data: HashMap<Key, Value>,
}

impl LuaTable {
    pub fn new() -> LuaTable {
        LuaTable {
            data: HashMap::new(),
        }
    }

    /// `t[k] = v`. A nil/UNRESOLVED value removes the key (Lua rule), so absent
    /// and nil are indistinguishable - exactly the Python model.
    pub fn set(&mut self, key: Key, value: Value) {
        match value {
            Value::Nil | Value::Unresolved => {
                self.data.remove(&key);
            }
            _ => {
                self.data.insert(key, value);
            }
        }
    }

    /// `t[k]` - an absent key is Lua nil (`Value::Nil`), NOT UNRESOLVED. The
    /// table is a resolved value, so `t[absent] or x` evaluates x.
    pub fn get(&self, key: &Key) -> Value {
        self.data.get(key).cloned().unwrap_or(Value::Nil)
    }

    /// Append at the array border (`table.insert(t, v)`).
    pub fn append(&mut self, value: Value) {
        let n = self.length();
        self.set(Key::Int(n + 1), value);
    }

    /// Lua `#t` / table.getn: the border of the 1..n contiguous array run.
    pub fn length(&self) -> i64 {
        let mut n = 0i64;
        while self.data.contains_key(&Key::Int(n + 1)) {
            n += 1;
        }
        n
    }

    /// `table.insert(t, pos, v)` - shift the array run up from `pos`.
    pub fn insert_at(&mut self, pos: i64, value: Value) {
        let n = self.length();
        let mut i = n;
        while i >= pos {
            let below = self.get(&Key::Int(i));
            self.set(Key::Int(i + 1), below);
            i -= 1;
        }
        self.set(Key::Int(pos), value);
    }

    /// `table.remove(t[, pos])` - remove and return, shifting the run down.
    pub fn remove(&mut self, pos: Option<i64>) -> Value {
        let n = self.length();
        if n == 0 {
            return Value::Nil;
        }
        let pos = pos.unwrap_or(n);
        let value = self.get(&Key::Int(pos));
        let mut i = pos;
        while i < n {
            let above = self.get(&Key::Int(i + 1));
            self.set(Key::Int(i), above);
            i += 1;
        }
        self.set(Key::Int(n), Value::Nil);
        value
    }

    /// `ipairs`: 1..border contiguous (i, v) pairs.
    pub fn ipairs(&self) -> Vec<(i64, Value)> {
        let mut out = Vec::new();
        let mut i = 1i64;
        loop {
            match self.data.get(&Key::Int(i)) {
                Some(v) => out.push((i, v.clone())),
                None => break,
            }
            i += 1;
        }
        out
    }

    /// `pairs`: every (key, value). Order is unspecified in Lua (and here); the
    /// interpreter only relies on ipairs order for the array part.
    pub fn pairs(&self) -> Vec<(Value, Value)> {
        self.data
            .iter()
            .map(|(k, v)| {
                let key = match k {
                    Key::Int(i) => Value::Num(*i as f64),
                    Key::Str(s) => Value::str(s.clone()),
                };
                (key, v.clone())
            })
            .collect()
    }
}
