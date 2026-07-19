//! The residue tick-loop interpreter - the HOT path, ported to native. Runs the
//! marshalled AST over the core-owned scope/tables/builtins, mirroring Python
//! `frame_eval.Interpreter` semantics exactly (keyframe_diff is the gate).
//!
//! Step 1 (this file) covers the PURE-LOGIC subset: literals, arithmetic,
//! control flow, locals/globals/accumulators, own tables + `table.*`/`type`/
//! `tonumber`/`tostring`/`math.*`, closures, and the Lua and/or/truthiness +
//! UNRESOLVED discipline. Anything that reads/writes the LIVE ENGINE (a `Method`
//! read, a poke, an index of a host handle) routes through the `Frontier` trait
//! (crate::frontier) - Python-implemented today. With the NoFrontier stub those
//! calls yield UNRESOLVED / no-op, so a pure-logic body runs fully native.

use std::rc::Rc;

use crate::ast::{Expr, Stmt};
use crate::builtins;
use crate::frontier::Frontier;
use crate::scope::{GlobalStore, ScopeArena};
use crate::table::{Key, LuaTable};
use crate::value::{binary, lua_and, lua_or, unary, Callable, Value};
use std::cell::RefCell;

/// Cap loops so a pathological literal loop cannot spin (matches Python
/// `_MAX_LOOP`); depth cap guards runaway recursion (`_MAX_DEPTH`).
const MAX_LOOP: i64 = 100_000;
const MAX_DEPTH: u32 = 200;

/// An interpreter closure: params + body + the scope id it captured. Called by
/// re-entering the interpreter with a fresh child scope.
pub struct Closure {
    pub params: Vec<Rc<str>>,
    pub body: Rc<[Stmt]>,
    pub defining_scope: usize,
}

/// Non-local control flow out of a body (a `return`).
enum Flow {
    Normal,
    Return(Value),
}

pub struct Interp<'a> {
    pub scopes: ScopeArena,
    pub globals: &'a mut dyn GlobalStore,
    pub frontier: &'a mut dyn Frontier,
    /// Names the body NEVER assigns to and that resolve to a host DATA TABLE:
    /// safe to snapshot into a native table once (read-only), eliminating the
    /// per-element `v[i][j]` frontier crossings. Empty when snapshotting is off.
    pub snapshottable: &'a std::collections::HashSet<String>,
    /// name -> the snapshotted native table, persisted across ticks (owned by
    /// the persistent NativeInterpreter). A name maps to Unresolved once probed
    /// and found NOT snapshottable, so it is not re-probed every tick.
    pub snapshots: &'a mut std::collections::HashMap<String, Value>,
    /// Per-tick DRIVER cache: `mod_time`/`beat`/... seeded once per tick by the
    /// sim (values it owns, the body never writes). Checked BEFORE the frontier
    /// so these hot reads (mod_time was 134 crossings/tick on gat) stay native.
    pub tick_cache: &'a std::collections::HashMap<String, Value>,
    /// Per-tick ACTOR-value cache: (handle, verb) -> the actor's current value,
    /// seeded by the sim after drain for the LEARNED read set. A GetX/GetY/... on
    /// a handle hits this instead of crossing.
    pub actor_cache: &'a std::collections::HashMap<(u64, String), Value>,
    /// The (handle, verb) actor reads this run made - recorded so the sim learns
    /// which values to seed next tick. `RefCell` so a `&self` read path can push.
    pub actor_reads: &'a std::cell::RefCell<Vec<(u64, String)>>,
}

impl<'a> Interp<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        globals: &'a mut dyn GlobalStore,
        frontier: &'a mut dyn Frontier,
        snapshottable: &'a std::collections::HashSet<String>,
        snapshots: &'a mut std::collections::HashMap<String, Value>,
        tick_cache: &'a std::collections::HashMap<String, Value>,
        actor_cache: &'a std::collections::HashMap<(u64, String), Value>,
        actor_reads: &'a std::cell::RefCell<Vec<(u64, String)>>,
    ) -> Interp<'a> {
        Interp {
            scopes: ScopeArena::new(),
            globals,
            frontier,
            snapshottable,
            snapshots,
            tick_cache,
            actor_cache,
            actor_reads,
        }
    }

    /// Run a body once at the persistent ROOT scope (top-level globals persist
    /// across calls; transient block scopes are truncated after).
    pub fn run(&mut self, body: &[Stmt]) {
        let keep = 1; // keep the root; drop nothing below it
        let _ = self.exec_block(body, ScopeArena::ROOT, 0);
        self.scopes.truncate(keep.max(1));
    }

    /// Read `name` as a GLOBAL - from the frontier's host env when it backs
    /// globals (so an accumulator shares the load-populated namespace + the
    /// guards read what the body wrote), else the core's own MapStore. Returns
    /// (found, value); UNRESOLVED value means unbound.
    fn global_lookup(&mut self, name: &str) -> (bool, Value) {
        // Per-tick driver clocks (mod_time/beat) are seeded native each tick;
        // hit them here instead of crossing to the host env every read.
        if let Some(v) = self.tick_cache.get(name) {
            return (true, v.clone());
        }
        if self.frontier.backs_globals() {
            let v = self.frontier.global_get(name);
            (!v.is_unresolved(), v)
        } else {
            (self.globals.has(name), self.globals.get(name))
        }
    }

    /// Write `name` as a global - to the frontier's host env or the MapStore.
    fn global_set(&mut self, name: &str, value: &Value) {
        if self.frontier.backs_globals() {
            self.frontier.global_set(name, value);
        } else {
            self.globals.set(name, value.clone());
        }
    }

    fn exec_block(&mut self, body: &[Stmt], scope: usize, depth: u32) -> Flow {
        for stmt in body {
            match self.exec(stmt, scope, depth) {
                Flow::Normal => {}
                ret @ Flow::Return(_) => return ret,
            }
            // A host-fn call that RAISED aborts the whole body tick (the Lua
            // path lets the error propagate to the fault handler); stop here so
            // native does not run statements Python never reached.
            if self.frontier.aborted() {
                return Flow::Normal;
            }
        }
        Flow::Normal
    }

    fn exec(&mut self, stmt: &Stmt, scope: usize, depth: u32) -> Flow {
        match stmt {
            Stmt::Local { names, values } => {
                let vals: Vec<Value> = values.iter().map(|v| self.eval(v, scope, depth)).collect();
                for (i, name) in names.iter().enumerate() {
                    let v = vals.get(i).cloned().unwrap_or(Value::Unresolved);
                    self.scopes.set_local(scope, name, v);
                }
                Flow::Normal
            }
            Stmt::Assign { targets, values } => {
                let vals: Vec<Value> = values.iter().map(|v| self.eval(v, scope, depth)).collect();
                for (i, target) in targets.iter().enumerate() {
                    let v = vals.get(i).cloned().unwrap_or(Value::Unresolved);
                    self.assign_target(target, v, scope, depth);
                }
                Flow::Normal
            }
            Stmt::If {
                cond,
                body,
                elifs,
                orelse,
            } => {
                if self.eval(cond, scope, depth).truthy() {
                    let child = self.scopes.child(scope);
                    return self.exec_block(body, child, depth);
                }
                for (econd, ebody) in elifs {
                    if self.eval(econd, scope, depth).truthy() {
                        let child = self.scopes.child(scope);
                        return self.exec_block(ebody, child, depth);
                    }
                }
                if !orelse.is_empty() {
                    let child = self.scopes.child(scope);
                    return self.exec_block(orelse, child, depth);
                }
                Flow::Normal
            }
            Stmt::NumericFor {
                var,
                start,
                stop,
                step,
                body,
            } => {
                self.exec_numeric_for(var, start, stop, step.as_ref(), body, scope, depth)
            }
            Stmt::GenericFor {
                names,
                exprs,
                body,
            } => self.exec_generic_for(names, exprs, body, scope, depth),
            Stmt::While { cond, body } => {
                let mut count = 0;
                while self.eval(cond, scope, depth).truthy() {
                    if count >= MAX_LOOP {
                        break;
                    }
                    let child = self.scopes.child(scope);
                    if let Flow::Return(v) = self.exec_block(body, child, depth) {
                        return Flow::Return(v);
                    }
                    count += 1;
                }
                Flow::Normal
            }
            Stmt::FuncDef {
                name,
                params,
                body,
            } => {
                let closure = self.make_closure(params.clone(), body.clone(), scope);
                self.assign_name(scope, name, closure);
                Flow::Normal
            }
            Stmt::Return { values } => {
                let v = values
                    .first()
                    .map(|e| self.eval(e, scope, depth))
                    .unwrap_or(Value::Nil);
                Flow::Return(v)
            }
            Stmt::ExprStmt(expr) => {
                self.exec_expr_stmt(expr, scope, depth);
                Flow::Normal
            }
            Stmt::Unparsed => Flow::Normal,
        }
    }

    fn exec_numeric_for(
        &mut self,
        var: &Rc<str>,
        start: &Expr,
        stop: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
        scope: usize,
        depth: u32,
    ) -> Flow {
        let s = self.eval(start, scope, depth).as_num();
        let e = self.eval(stop, scope, depth).as_num();
        let st = match step {
            Some(x) => self.eval(x, scope, depth).as_num(),
            None => Some(1.0),
        };
        let (s, e, st) = match (s, e, st) {
            (Some(s), Some(e), Some(st)) if st != 0.0 => (s, e, st),
            _ => return Flow::Normal,
        };
        let mut i = s;
        let mut count = 0i64;
        while (st > 0.0 && i <= e) || (st < 0.0 && i >= e) {
            if count >= MAX_LOOP {
                break;
            }
            let child = self.scopes.child(scope);
            self.scopes.set_local(child, var, Value::Num(i));
            if let Flow::Return(v) = self.exec_block(body, child, depth) {
                return Flow::Return(v);
            }
            i += st;
            count += 1;
        }
        Flow::Normal
    }

    fn exec_generic_for(
        &mut self,
        names: &[Rc<str>],
        exprs: &[Expr],
        body: &[Stmt],
        scope: usize,
        depth: u32,
    ) -> Flow {
        // ipairs(t) / pairs(t) over an own LuaTable; a host table iterates via
        // the frontier. Anything else -> skip (the tree-walk floor).
        let pairs = match self.iter_pairs(exprs, scope, depth) {
            Some(p) => p,
            None => return Flow::Normal,
        };
        for kv in pairs {
            let child = self.scopes.child(scope);
            for (i, name) in names.iter().enumerate() {
                let v = kv.get(i).cloned().unwrap_or(Value::Unresolved);
                self.scopes.set_local(child, name, v);
            }
            if let Flow::Return(v) = self.exec_block(body, child, depth) {
                return Flow::Return(v);
            }
        }
        Flow::Normal
    }

    /// Resolve `for .. in ipairs(t)/pairs(t)` to the (k, v) rows. Only ipairs /
    /// pairs over an own table are modeled natively; a host table defers to the
    /// frontier's `iter_table`.
    fn iter_pairs(&mut self, exprs: &[Expr], scope: usize, depth: u32) -> Option<Vec<Vec<Value>>> {
        let first = exprs.first()?;
        if let Expr::Call { fn_, args } = first {
            if let Expr::Sym(name) = &**fn_ {
                let arg = args.first()?;
                let table = self.eval(arg, scope, depth);
                return self.iter_over(name, table);
            }
        }
        None
    }

    fn iter_over(&mut self, kind: &str, table: Value) -> Option<Vec<Vec<Value>>> {
        match &table {
            Value::Table(t) => {
                let t = t.borrow();
                let rows = match kind {
                    "ipairs" => t
                        .ipairs()
                        .into_iter()
                        .map(|(i, v)| vec![Value::Num(i as f64), v])
                        .collect(),
                    "pairs" => t.pairs().into_iter().map(|(k, v)| vec![k, v]).collect(),
                    _ => return None,
                };
                Some(rows)
            }
            Value::Handle(id) => self.frontier.iter_table(*id),
            _ => None,
        }
    }

    fn exec_expr_stmt(&mut self, expr: &Expr, scope: usize, depth: u32) {
        // A method statement is an EFFECT (a poke); a plain call is a
        // free-function effect; anything else is evaluated for side effects.
        if let Expr::Method { recv, name, args } = expr {
            let recv_v = self.eval(recv, scope, depth);
            let arg_vs: Vec<Value> = args.iter().map(|a| self.eval(a, scope, depth)).collect();
            self.frontier.poke(&recv_v, name, &arg_vs);
        } else {
            let _ = self.eval(expr, scope, depth);
        }
    }

    /// `name = value`: rebind the nearest local, else write a global (to the
    /// frontier's host env or the MapStore). The Python `Scope.assign` rule.
    fn assign_name(&mut self, scope: usize, name: &str, value: Value) {
        if !self.scopes.assign_local(scope, name, &value) {
            self.global_set(name, &value);
        }
    }

    fn assign_target(&mut self, target: &Expr, value: Value, scope: usize, depth: u32) {
        match target {
            Expr::Sym(name) => {
                // `_G[k]` computed writes come through Index; a bare Sym assigns.
                self.assign_name(scope, name, value);
            }
            Expr::Index { base, key } => {
                // `_G['name'] = v` is a computed global write.
                if let Expr::Sym(bname) = &**base {
                    if &**bname == "_G" {
                        let k = self.eval(key, scope, depth);
                        if let Some(name) = key_as_name(&k) {
                            self.assign_name(scope, &name, value);
                        }
                        return;
                    }
                }
                let table = self.eval(base, scope, depth);
                let k = self.eval(key, scope, depth);
                self.assign_element(&table, &k, value);
            }
            Expr::Field { base, name } => {
                let table = self.eval(base, scope, depth);
                self.assign_element(&table, &Value::str(name.to_string()), value);
            }
            _ => {}
        }
    }

    fn assign_element(&mut self, table: &Value, key: &Value, value: Value) {
        match table {
            Value::Table(t) => {
                if let Some(k) = Key::from_value(key) {
                    t.borrow_mut().set(k, value);
                }
            }
            // A host table write goes through the frontier (set_index).
            Value::Handle(id) => {
                self.frontier.set_index(*id, key, &value);
            }
            _ => {}
        }
    }

    // -- expressions --------------------------------------------------------

    fn eval(&mut self, expr: &Expr, scope: usize, depth: u32) -> Value {
        if depth > MAX_DEPTH {
            return Value::Unresolved;
        }
        match expr {
            Expr::Num(n) => Value::Num(*n),
            Expr::Str(s) => Value::Str(s.clone()),
            Expr::Bool(b) => Value::Bool(*b),
            Expr::Nil => Value::Nil,
            Expr::Sym(name) => self.eval_symbol(name, scope),
            Expr::Index { base, key } => self.eval_index(base, key, scope, depth),
            Expr::Field { base, name } => self.eval_field(base, name, scope, depth),
            Expr::Unary { op, operand } => {
                let x = self.eval(operand, scope, depth);
                unary(op, x)
            }
            Expr::Binary { op, left, right } => self.eval_binary(op, left, right, scope, depth),
            Expr::Call { fn_, args } => self.eval_call(fn_, args, scope, depth),
            Expr::Method { recv, name, args } => self.eval_method(recv, name, args, scope, depth),
            Expr::Table { array, fields } => self.eval_table(array, fields, scope, depth),
            Expr::FuncExpr { params, body } => {
                self.make_closure(params.clone(), body.clone(), scope)
            }
            Expr::Unparsed => Value::Unresolved,
        }
    }

    fn eval_symbol(&mut self, name: &str, scope: usize) -> Value {
        if let Some(v) = self.scopes.lookup_local(scope, name) {
            return v;
        }
        // A snapshottable DATA TABLE: return the native copy (snapshotted once,
        // cached across ticks), so `v[i][j]` reads never cross the frontier.
        if let Some(v) = self.snapshot_lookup(name) {
            return v;
        }
        let (found, value) = self.global_lookup(name);
        if found {
            value
        } else {
            // Not a local/global: ask the frontier (a driver clock symbol like
            // `beat`, an actor global). Pure-logic path -> UNRESOLVED.
            self.frontier.symbol(name)
        }
    }

    /// If `name` is snapshottable, return its native table (snapshotting on
    /// first sight, caching thereafter). None when not snapshottable, so the
    /// caller falls through to the normal global/frontier read. A name that
    /// probes as non-snapshottable caches `Unresolved` so it is not re-probed.
    fn snapshot_lookup(&mut self, name: &str) -> Option<Value> {
        if !self.snapshottable.contains(name) {
            return None;
        }
        if let Some(v) = self.snapshots.get(name) {
            return match v {
                Value::Unresolved => None, // probed, not snapshottable
                other => Some(other.clone()),
            };
        }
        let snap = self.frontier.snapshot_global(name);
        self.snapshots.insert(name.to_string(), snap.clone());
        match snap {
            Value::Unresolved => None,
            other => Some(other),
        }
    }

    fn eval_binary(&mut self, op: &str, left: &Expr, right: &Expr, scope: usize, depth: u32) -> Value {
        match op {
            "and" => {
                let a = self.eval(left, scope, depth);
                lua_and(a, || self.eval(right, scope, depth))
            }
            "or" => {
                let a = self.eval(left, scope, depth);
                lua_or(a, || self.eval(right, scope, depth))
            }
            _ => {
                let a = self.eval(left, scope, depth);
                let b = self.eval(right, scope, depth);
                binary(op, a, b)
            }
        }
    }

    fn eval_index(&mut self, base: &Expr, key: &Expr, scope: usize, depth: u32) -> Value {
        // `_G[k]` computed global read.
        if let Expr::Sym(bname) = base {
            if &**bname == "_G" {
                let k = self.eval(key, scope, depth);
                if let Some(name) = key_as_name(&k) {
                    return self.eval_symbol(&name, scope);
                }
                return Value::Unresolved;
            }
        }
        let b = self.eval(base, scope, depth);
        let k = self.eval(key, scope, depth);
        if b.is_unresolved() || k.is_unresolved() {
            return Value::Unresolved;
        }
        self.index_value(&b, &k)
    }

    fn eval_field(&mut self, base: &Expr, name: &str, scope: usize, depth: u32) -> Value {
        // `math.pi`/`math.huge` fold to a literal.
        if let Expr::Sym(bname) = base {
            if &**bname == "math" {
                if let Some(c) = builtins::lua_math_const(name) {
                    return c;
                }
            }
        }
        let b = self.eval(base, scope, depth);
        if b.is_unresolved() {
            return Value::Unresolved;
        }
        self.index_value(&b, &Value::str(name.to_string()))
    }

    /// `base[key]` for a resolved base+key. Own table reads natively; a host
    /// handle defers to the frontier, whose UNRESOLVED (absent-but-resolved)
    /// maps to nil (the resolved-nil rule).
    fn index_value(&mut self, base: &Value, key: &Value) -> Value {
        match base {
            Value::Table(t) => match Key::from_value(key) {
                Some(k) => t.borrow().get(&k),
                None => Value::Nil,
            },
            Value::Handle(id) => resolved_nil(self.frontier.index(*id, key)),
            _ => Value::Unresolved,
        }
    }

    fn eval_call(&mut self, fn_: &Expr, args: &[Expr], scope: usize, depth: u32) -> Value {
        let arg_vs: Vec<Value> = args.iter().map(|a| self.eval(a, scope, depth)).collect();
        // Stdlib builtins over own values (`table.*`, `type`, `tonumber`,
        // `tostring`, `math.*`) - handled natively so no host round-trip.
        if let Some(v) = self.builtin_call(fn_, &arg_vs) {
            return v;
        }
        match fn_ {
            Expr::Sym(name) => {
                let local = self.scopes.lookup_local(scope, name);
                let (found, bound) = match local {
                    Some(v) => (true, v),
                    None => self.global_lookup(name),
                };
                if found {
                    if let Value::Func(c) = &bound {
                        return self.call_callable(c.clone(), &arg_vs, depth);
                    }
                }
                if !found {
                    // A global host function (`SecondsToClock`) resolves via the
                    // frontier symbol; if callable, call it there.
                    let global_fn = self.frontier.symbol(name);
                    if let Value::Func(c) = &global_fn {
                        return self.call_callable(c.clone(), &arg_vs, depth);
                    }
                    if global_fn.is_unresolved() && arg_vs.iter().any(|a| a.is_unresolved()) {
                        return Value::Unresolved;
                    }
                }
                if arg_vs.iter().any(|a| a.is_unresolved()) {
                    return Value::Unresolved;
                }
                self.frontier.call(name, &arg_vs)
            }
            _ => {
                let target = self.eval(fn_, scope, depth);
                if let Value::Func(c) = &target {
                    self.call_callable(c.clone(), &arg_vs, depth)
                } else {
                    Value::Unresolved
                }
            }
        }
    }

    /// `table.*`/`type`/`tonumber`/`tostring`/`math.*` over own values. Returns
    /// None when `fn_` is not such a builtin (flows on to closures/frontier).
    fn builtin_call(&mut self, fn_: &Expr, args: &[Value]) -> Option<Value> {
        match fn_ {
            Expr::Sym(name) => match &**name {
                "type" => args.first().map(|v| Value::str(builtins::lua_type(v))),
                "tonumber" => args
                    .first()
                    .map(|v| builtins::lua_tonumber(v, args.get(1))),
                "tostring" => args.first().map(|v| Value::str(builtins::lua_tostring(v))),
                _ => None,
            },
            Expr::Field { base, name } => {
                if let Expr::Sym(bn) = &**base {
                    match &**bn {
                        "table" => self.table_builtin(name, args),
                        "math" => builtins::lua_math(name, args),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn table_builtin(&mut self, name: &str, args: &[Value]) -> Option<Value> {
        let table = match args.first() {
            Some(Value::Table(t)) => t.clone(),
            _ => return None, // a host table handles its own upstream
        };
        match name {
            "insert" => {
                let mut t = table.borrow_mut();
                if args.len() >= 3 {
                    if let Some(pos) = args[1].as_num() {
                        t.insert_at(pos as i64, args[2].clone());
                    }
                } else if let Some(v) = args.get(1) {
                    t.append(v.clone());
                }
                Some(Value::Nil)
            }
            "remove" => {
                let pos = args.get(1).and_then(|v| v.as_num()).map(|n| n as i64);
                Some(table.borrow_mut().remove(pos))
            }
            "getn" => Some(Value::Num(table.borrow().length() as f64)),
            _ => None,
        }
    }

    fn eval_method(&mut self, recv: &Expr, name: &str, args: &[Expr], scope: usize, depth: u32) -> Value {
        let recv_v = self.eval(recv, scope, depth);
        if recv_v.is_unresolved() {
            return Value::Unresolved;
        }
        // Actor-value getter (GetX/GetY/GetZ/getaux/GetText) on a handle: hit the
        // per-tick actor cache the sim seeded after drain, so these ~23 reads/
        // tick stay native. The verb is RECORDED so the sim knows which
        // (actor,verb) to seed next tick (learn-then-cache). A miss falls to the
        // frontier (first tick, or a verb/actor not yet learned).
        if let Value::Handle(id) = &recv_v {
            if args.is_empty() && is_actor_getter(name) {
                self.actor_reads.borrow_mut().push((*id, name.to_string()));
                if let Some(v) = self.actor_cache.get(&(*id, name.to_string())) {
                    return v.clone();
                }
            }
        }
        let arg_vs: Vec<Value> = args.iter().map(|a| self.eval(a, scope, depth)).collect();
        // A method in VALUE position is a getter read through the frontier.
        self.frontier.method(&recv_v, name, &arg_vs)
    }

    fn eval_table(&mut self, array: &[Expr], fields: &[(Rc<str>, Expr)], scope: usize, depth: u32) -> Value {
        let mut table = LuaTable::new();
        for item in array {
            let v = self.eval(item, scope, depth);
            table.append(v);
        }
        for (k, v) in fields {
            let value = self.eval(v, scope, depth);
            table.set(Key::Str(k.to_string()), value);
        }
        Value::Table(Rc::new(RefCell::new(table)))
    }

    fn make_closure(&mut self, params: Vec<Rc<str>>, body: Rc<[Stmt]>, scope: usize) -> Value {
        Value::Func(Callable::Closure(Rc::new(Closure {
            params,
            body,
            defining_scope: scope,
        })))
    }

    fn call_callable(&mut self, c: Callable, args: &[Value], depth: u32) -> Value {
        if depth > MAX_DEPTH {
            return Value::Unresolved;
        }
        match c {
            Callable::Closure(closure) => {
                let call_scope = self.scopes.child(closure.defining_scope);
                for (i, param) in closure.params.iter().enumerate() {
                    let v = args.get(i).cloned().unwrap_or(Value::Unresolved);
                    self.scopes.set_local(call_scope, param, v);
                }
                match self.exec_block(&closure.body, call_scope, depth + 1) {
                    Flow::Return(v) => v,
                    Flow::Normal => Value::Nil,
                }
            }
            // A host callable is invoked across the frontier (it marshals
            // UNRESOLVED->nil so a host fn never does arithmetic on the sentinel).
            Callable::Host(id) => self.frontier.call_host(id, args),
        }
    }
}

/// Verbs that read an actor's current animated value (no args) - the ones the
/// sim can seed into the per-tick actor cache. GetSongBeat is NOT here (it is a
/// singleton clock, handled by the beat driver cache).
fn is_actor_getter(name: &str) -> bool {
    matches!(
        name,
        "GetX" | "GetY" | "GetZ" | "getaux" | "GetZoom" | "GetZoomX" | "GetZoomY"
    )
}

/// A computed `_G[k]` name: only a string/number key names a global.
fn key_as_name(k: &Value) -> Option<String> {
    match k {
        Value::Str(s) => Some(s.to_string()),
        Value::Num(n) => Some(crate::value::num_to_lua_string(*n)),
        _ => None,
    }
}

/// Map a frontier read of a RESOLVED host base+key from UNRESOLVED to nil - an
/// absent host-table element is a KNOWN nil (the resolved-nil rule), not an
/// unprovable read.
fn resolved_nil(v: Value) -> Value {
    if v.is_unresolved() {
        Value::Nil
    } else {
        v
    }
}
