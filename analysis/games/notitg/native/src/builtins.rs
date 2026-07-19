//! Lua stdlib the core services itself - value-inspection globals
//! (`type`/`tonumber`/`tostring`), `table.*` over the core's own `LuaTable`, and
//! native `math.*` inlined (no host call). Mirrors the Python `frame_eval`
//! builtins and `compile_sched._MATH` set; the core owns math/stdlib so the
//! measured 7.6M per-tick data-table+math crossings never leave Rust.

use crate::value::{num_to_lua_string, Value};

/// Lua `type(v)` -> the type name. A host table vs a host function is told apart
/// by the frontier upstream (the core sees a distinct `Table`/`Func`/`Handle`);
/// here nil/UNRESOLVED is 'nil', primitives map directly.
pub fn lua_type(v: &Value) -> &'static str {
    match v {
        Value::Nil | Value::Unresolved => "nil",
        Value::Bool(_) => "boolean",
        Value::Num(_) => "number",
        Value::Str(_) => "string",
        Value::Table(_) => "table",
        Value::Func(_) => "function",
        // A host handle is a table today (actor/lupa table); a host FUNCTION is
        // marshalled as Func by the frontier, so a Handle is a table.
        Value::Handle(_) => "table",
    }
}

/// Lua `tonumber(v[, base])`. Number passes through; a numeric string parses
/// (optional integer base). nil/UNRESOLVED/table/bool -> nil. The Python-side
/// `inf`/`nan` word rejection is matched (Lua tonumber does not parse them).
pub fn lua_tonumber(v: &Value, base: Option<&Value>) -> Value {
    match v {
        Value::Num(n) => Value::Num(*n),
        Value::Str(s) => parse_number(s, base),
        _ => Value::Nil,
    }
}

fn parse_number(s: &str, base: Option<&Value>) -> Value {
    let trimmed = s.trim();
    if let Some(b) = base.and_then(|b| b.as_num()) {
        return match i64::from_str_radix(trimmed.trim_start_matches(['+', '-']), b as u32) {
            Ok(mag) => {
                let signed = if trimmed.starts_with('-') { -mag } else { mag };
                Value::Num(signed as f64)
            }
            Err(_) => Value::Nil,
        };
    }
    let lower = trimmed.trim_start_matches(['+', '-']).to_ascii_lowercase();
    if lower == "inf" || lower == "nan" || lower == "infinity" {
        return Value::Nil; // Lua tonumber does not parse these words
    }
    // Hex literal (Lua accepts 0x..); else a plain decimal/float.
    if lower.starts_with("0x") {
        let hexbody = trimmed.trim_start_matches(['+', '-']).trim_start_matches("0x");
        return match i64::from_str_radix(hexbody, 16) {
            Ok(mag) => {
                let signed = if trimmed.starts_with('-') { -mag } else { mag };
                Value::Num(signed as f64)
            }
            Err(_) => Value::Nil,
        };
    }
    // A genuine `1e400` overflows to inf (as Lua does); the `inf` WORD was
    // already rejected above, so any inf here came from numeric overflow.
    match trimmed.parse::<f64>() {
        Ok(n) => Value::Num(n),
        Err(_) => Value::Nil,
    }
}

/// Lua `tostring(v)`. nil/UNRESOLVED -> "nil", bool -> "true"/"false", an
/// integer float drops the `.0`.
pub fn lua_tostring(v: &Value) -> String {
    match v {
        Value::Nil | Value::Unresolved => "nil".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Num(n) => num_to_lua_string(*n),
        Value::Str(s) => s.to_string(),
        _ => "table".to_string(),
    }
}

/// The `math.*` functions inlined as native ops (the analytic curve set). Pure
/// deterministic only - `math.random` is intentionally absent (handled by the
/// frontier / stays residue). Returns None for a name we do not model, so the
/// caller falls through.
pub fn lua_math(name: &str, args: &[Value]) -> Option<Value> {
    let a = args.first().and_then(|v| v.as_num());
    let b = args.get(1).and_then(|v| v.as_num());
    let one = |f: fn(f64) -> f64| a.map(|x| Value::Num(f(x))).or(Some(Value::Unresolved));
    match name {
        "sin" => one(f64::sin),
        "cos" => one(f64::cos),
        "tan" => one(f64::tan),
        "asin" => one(f64::asin),
        "acos" => one(f64::acos),
        "atan" => one(f64::atan),
        "sinh" => one(f64::sinh),
        "cosh" => one(f64::cosh),
        "tanh" => one(f64::tanh),
        "exp" => one(f64::exp),
        "log" => one(f64::ln),
        "log10" => one(f64::log10),
        "sqrt" => one(f64::sqrt),
        "abs" => one(f64::abs),
        "floor" => one(f64::floor),
        "ceil" => one(f64::ceil),
        "deg" => one(f64::to_degrees),
        "rad" => one(f64::to_radians),
        "atan2" => Some(bin(a, b, f64::atan2)),
        "fmod" => Some(bin(a, b, |x, y| x % y)),
        "pow" => Some(bin(a, b, f64::powf)),
        "min" => Some(fold(args, f64::min)),
        "max" => Some(fold(args, f64::max)),
        _ => None,
    }
}

fn bin(a: Option<f64>, b: Option<f64>, f: fn(f64, f64) -> f64) -> Value {
    match (a, b) {
        (Some(x), Some(y)) => Value::Num(f(x, y)),
        _ => Value::Unresolved,
    }
}

fn fold(args: &[Value], f: fn(f64, f64) -> f64) -> Value {
    let mut acc: Option<f64> = None;
    for v in args {
        match v.as_num() {
            Some(n) => acc = Some(acc.map_or(n, |a| f(a, n))),
            None => return Value::Unresolved,
        }
    }
    acc.map(Value::Num).unwrap_or(Value::Unresolved)
}

/// `math.pi`/`math.huge` constants that fold to a literal.
pub fn lua_math_const(name: &str) -> Option<Value> {
    match name {
        "pi" => Some(Value::Num(std::f64::consts::PI)),
        "huge" => Some(Value::Num(f64::INFINITY)),
        _ => None,
    }
}
