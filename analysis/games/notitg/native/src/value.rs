//! The core value model - number / string / bool / nil / array / func /
//! UNRESOLVED, with the source language's operand semantics (Lua `and`/`or`
//! return-operand rules, truthiness) and the "skip, do not guess" UNRESOLVED
//! discipline. This mirrors the Python `frame_eval` reference oracle exactly;
//! keyframe_diff gates the equality.
//!
//! `Unresolved` is the poison sentinel: any arithmetic/comparison with an
//! UNRESOLVED operand yields UNRESOLVED, with the short-circuit exceptions for
//! `and`/`or`. A `Nil` is a KNOWN absence (Lua nil), distinct from UNRESOLVED
//! (an unprovable read) - the distinction is load-bearing for `and`/`or`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::table::LuaTable;

/// A callable the core holds: an interpreter closure (own body) or a host
/// function proxied from the frontier. The core only stores/passes it; calling
/// is the interpreter's job (closures) or the frontier's (host fns).
///
/// `Host` and the `Handle` value variant are the step-2 frontier seam (a live
/// lupa function/actor id); step 1's pure-logic path never constructs them.
#[derive(Clone)]
#[allow(dead_code)] // Host: constructed by the step-2 frontier marshalling
pub enum Callable {
    /// An interpreter-made closure: captured scope + params + body index.
    Closure(Rc<crate::eval::Closure>),
    /// A host callable behind the frontier, addressed by an opaque id the
    /// frontier resolves (a lupa function today). The core never calls it
    /// directly - it hands the id back across the frontier.
    Host(u64),
}

/// The core value. `Array` is the growable-vec table model the corpus showed is
/// enough (0 hash-keyed constructors); a mixed/hash table uses `Table`.
#[derive(Clone)]
#[allow(dead_code)] // Handle: constructed by the step-2 frontier marshalling
pub enum Value {
    Num(f64),
    Str(Rc<str>),
    Bool(bool),
    Nil,
    Table(Rc<RefCell<LuaTable>>),
    Func(Callable),
    /// A HOST handle (an actor id / lupa table id) the frontier owns; opaque to
    /// the core, passed back across the frontier for reads/pokes/index.
    Handle(u64),
    /// The skip-don't-guess sentinel (Python `UNRESOLVED`). NOT the same as
    /// `Nil`: an unprovable read, not a known absence.
    Unresolved,
}

impl Value {
    pub fn str(s: impl Into<Rc<str>>) -> Value {
        Value::Str(s.into())
    }

    pub fn is_unresolved(&self) -> bool {
        matches!(self, Value::Unresolved)
    }

    /// Lua truthiness with the UNRESOLVED discipline for control flow: an
    /// unprovable condition is FALSE (skip, do not guess). Only nil and false
    /// are falsy; 0 and "" are TRUE.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Unresolved | Value::Nil | Value::Bool(false) => false,
            _ => true,
        }
    }

    /// Lua truthiness IGNORING the UNRESOLVED rule - for `and`/`or` operand
    /// selection, which decides on the RAW operand (callers guard UNRESOLVED
    /// before reaching here).
    pub fn truthy_raw(&self) -> bool {
        !matches!(self, Value::Nil | Value::Bool(false))
    }

    /// The numeric value if this is a plain number (not a bool), else None -
    /// the `_num` helper (bool is not a number in Lua arithmetic here).
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }
}

/// Lua `a and b`: returns the OPERAND. `a` when `a` is falsy, else `b`. An
/// UNRESOLVED `a` makes the choice unknowable -> UNRESOLVED (the right operand
/// cannot rescue it).
pub fn lua_and(a: Value, right: impl FnOnce() -> Value) -> Value {
    if a.is_unresolved() {
        return Value::Unresolved;
    }
    if a.truthy_raw() {
        right()
    } else {
        a
    }
}

/// Lua `a or b`: `a` when truthy, else `b`. UNRESOLVED `a` -> UNRESOLVED.
pub fn lua_or(a: Value, right: impl FnOnce() -> Value) -> Value {
    if a.is_unresolved() {
        return Value::Unresolved;
    }
    if a.truthy_raw() {
        a
    } else {
        right()
    }
}

/// Unary `-`/`not`/`#`. `-x` on a non-number is UNRESOLVED (the interpreter's
/// rule); `not` is raw-truthiness; `#` is length of a table/string.
pub fn unary(op: &str, x: Value) -> Value {
    if x.is_unresolved() {
        return Value::Unresolved;
    }
    match op {
        "-" => match x {
            Value::Num(n) => Value::Num(-n),
            _ => Value::Unresolved,
        },
        "not" => Value::Bool(!x.truthy_raw()),
        "#" => match &x {
            Value::Str(s) => Value::Num(s.chars().count() as f64),
            Value::Table(t) => Value::Num(t.borrow().length() as f64),
            _ => Value::Unresolved,
        },
        _ => Value::Unresolved,
    }
}

/// Binary arithmetic / comparison / concat. Any UNRESOLVED operand -> UNRESOLVED
/// (except `..` which still concats known operands only). A type error or divide
/// by zero yields UNRESOLVED, matching the Python `_binary` try/except.
pub fn binary(op: &str, a: Value, b: Value) -> Value {
    if a.is_unresolved() || b.is_unresolved() {
        return Value::Unresolved;
    }
    if op == ".." {
        return match (concat_str(&a), concat_str(&b)) {
            (Some(x), Some(y)) => Value::str(format!("{x}{y}")),
            _ => Value::Unresolved,
        };
    }
    // Equality / inequality work across types (Lua `==`); the rest need numbers.
    match op {
        "==" => return Value::Bool(value_eq(&a, &b)),
        "~=" => return Value::Bool(!value_eq(&a, &b)),
        _ => {}
    }
    let (x, y) = match (a.as_num(), b.as_num()) {
        (Some(x), Some(y)) => (x, y),
        _ => return Value::Unresolved,
    };
    match op {
        "+" => Value::Num(x + y),
        "-" => Value::Num(x - y),
        "*" => Value::Num(x * y),
        "/" => {
            if y == 0.0 {
                Value::Unresolved
            } else {
                Value::Num(x / y)
            }
        }
        "%" => {
            if y == 0.0 {
                Value::Unresolved
            } else {
                // Lua `%` is floored modulo (a - floor(a/b)*b), not Rust's rem.
                Value::Num(x - (x / y).floor() * y)
            }
        }
        "^" => Value::Num(x.powf(y)),
        "<" => Value::Bool(x < y),
        "<=" => Value::Bool(x <= y),
        ">" => Value::Bool(x > y),
        ">=" => Value::Bool(x >= y),
        _ => Value::Unresolved,
    }
}

/// Lua `==`: numbers/strings/bools compare by value; nil == nil; everything
/// else (tables/funcs/handles) by identity, which we approximate as never-equal
/// unless the same Rc (the interpreter rarely compares tables for equality).
fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (Value::Handle(x), Value::Handle(y)) => x == y,
        (Value::Table(x), Value::Table(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// The `..` string form of a value: a number prints Lua-style (an integer float
/// drops the `.0`), a string is itself. nil/bool/table are not concatenable ->
/// None -> the whole concat is UNRESOLVED.
fn concat_str(v: &Value) -> Option<String> {
    match v {
        Value::Str(s) => Some(s.to_string()),
        Value::Num(n) => Some(num_to_lua_string(*n)),
        _ => None,
    }
}

/// Lua number->string: an integer-valued float prints without the `.0`
/// (`2`, not `2.0`), matching Python `_concat`.
pub fn num_to_lua_string(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}
