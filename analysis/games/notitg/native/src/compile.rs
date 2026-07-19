//! Closure compiler for the frame interpreter - the Rust analogue of Python's
//! `frame_compile_exec`. The tree-walk `eval`/`exec` re-dispatches on node type
//! (a `match`) at EVERY node EVERY tick; profiling showed that dispatch is the
//! gat bottleneck (1.3s of a 2.28s window), NOT the frontier crossings. This
//! compiles a body's AST ONCE into nested closures: each node becomes an
//! `Fn(&mut Interp) -> Value` (expr) / `-> Flow` (stmt) closing over its
//! children's compiled closures, so per-tick execution is a direct call chain -
//! no per-node match, no AST re-walk.
//!
//! Semantics are IDENTICAL to `eval`/`exec` (keyframe_diff gates it); a node
//! shape outside the compiled subset defers to the tree-walk (`eval_node`/
//! `exec_node`), so the compiler is a strict speedup, never a coverage
//! regression. The `for<'a>` bound lets a compiled closure be stored on the
//! persistent NativeInterpreter yet run against any tick's borrowed `Interp<'a>`.

use std::rc::Rc;

use crate::ast::{Expr, Stmt};
use crate::eval::{Flow, Interp};
use crate::value::{binary, lua_and, lua_or, unary, Value};

pub type CExpr = Rc<dyn for<'a> Fn(&mut Interp<'a>, usize, u32) -> Value>;
pub type CStmt = Rc<dyn for<'a> Fn(&mut Interp<'a>, usize, u32) -> Flow>;

/// Compile a statement sequence into one closure running them in order, stopping
/// on a Return or a frontier abort (the `exec_block` contract).
pub fn compile_block(body: &[Stmt]) -> CStmt {
    let compiled: Vec<CStmt> = body.iter().map(compile_stmt).collect();
    Rc::new(move |interp, scope, depth| {
        for c in &compiled {
            match c(interp, scope, depth) {
                Flow::Normal => {}
                ret @ Flow::Return(_) => return ret,
            }
            if interp.frontier_aborted() {
                return Flow::Normal;
            }
        }
        Flow::Normal
    })
}

// -- expressions -------------------------------------------------------------

pub fn compile_expr(node: &Expr) -> CExpr {
    match node {
        Expr::Num(n) => {
            let n = *n;
            Rc::new(move |_, _, _| Value::Num(n))
        }
        Expr::Str(s) => {
            let s = s.clone();
            Rc::new(move |_, _, _| Value::Str(s.clone()))
        }
        Expr::Bool(b) => {
            let b = *b;
            Rc::new(move |_, _, _| Value::Bool(b))
        }
        Expr::Nil => Rc::new(|_, _, _| Value::Nil),
        Expr::Sym(name) => {
            let name: String = name.to_string();
            Rc::new(move |interp, scope, _| interp.read_symbol(&name, scope))
        }
        Expr::Unary { op, operand } => {
            let op: String = op.to_string();
            let operand = compile_expr(operand);
            Rc::new(move |interp, scope, depth| {
                let x = operand(interp, scope, depth);
                unary(&op, x)
            })
        }
        Expr::Binary { op, left, right } => compile_binary(op, left, right),
        // Index/Field/Call/Method/Table/FuncExpr touch the frontier, own tables,
        // or closures; defer to the tree-walk `eval_node` (still correct). These
        // are the minority of evals; the arithmetic/symbol spine above is the hot
        // part the closure form accelerates.
        other => {
            let node = other.clone();
            Rc::new(move |interp, scope, depth| interp.eval_node(&node, scope, depth))
        }
    }
}

fn compile_binary(op: &str, left: &Expr, right: &Expr) -> CExpr {
    let l = compile_expr(left);
    let r = compile_expr(right);
    let op: String = op.to_string();
    match op.as_str() {
        "and" => Rc::new(move |interp, scope, depth| {
            let a = l(interp, scope, depth);
            lua_and(a, || r(interp, scope, depth))
        }),
        "or" => Rc::new(move |interp, scope, depth| {
            let a = l(interp, scope, depth);
            lua_or(a, || r(interp, scope, depth))
        }),
        _ => Rc::new(move |interp, scope, depth| {
            let a = l(interp, scope, depth);
            let b = r(interp, scope, depth);
            binary(&op, a, b)
        }),
    }
}

// -- statements --------------------------------------------------------------

pub fn compile_stmt(node: &Stmt) -> CStmt {
    match node {
        Stmt::ExprStmt(expr) => compile_expr_stmt(expr),
        Stmt::If {
            cond,
            body,
            elifs,
            orelse,
        } => compile_if(cond, body, elifs, orelse),
        // Assign/Local/For/While/FuncDef/Return: defer to the tree-walk `exec`
        // (they mutate scope/tables; the tree-walk handles them and the frontier
        // abort check in compile_block still applies between statements).
        other => {
            let node = other.clone();
            Rc::new(move |interp, scope, depth| interp.exec_node(&node, scope, depth))
        }
    }
}

/// A bare `ExprStmt`: a poke (`recv:verb(args)`) is an effect the tree-walk
/// `exec` routes to the frontier; a plain call is evaluated for side effects.
/// We keep the whole statement on the tree-walk (poke handling lives there), so
/// this just forwards - but compiling the surrounding block still removed the
/// per-node match for its expression children where they are arithmetic.
fn compile_expr_stmt(expr: &Expr) -> CStmt {
    match expr {
        // A method call in statement position is a poke - tree-walk owns it.
        Expr::Method { .. } => {
            let node = Stmt::ExprStmt(expr.clone());
            Rc::new(move |interp, scope, depth| interp.exec_node(&node, scope, depth))
        }
        other => {
            let compiled = compile_expr(other);
            Rc::new(move |interp, scope, depth| {
                compiled(interp, scope, depth);
                Flow::Normal
            })
        }
    }
}

/// `if cond then body [elseif...] [else...]` - the condition and branch bodies
/// are compiled; each branch runs in a fresh child scope (matching `exec`).
fn compile_if(
    cond: &Expr,
    body: &[Stmt],
    elifs: &[(Expr, Rc<[Stmt]>)],
    orelse: &[Stmt],
) -> CStmt {
    let cond = compile_expr(cond);
    let then_body = compile_block(body);
    let elifs: Vec<(CExpr, CStmt)> = elifs
        .iter()
        .map(|(c, b)| (compile_expr(c), compile_block(b)))
        .collect();
    let orelse = if orelse.is_empty() {
        None
    } else {
        Some(compile_block(orelse))
    };
    Rc::new(move |interp, scope, depth| {
        if cond(interp, scope, depth).truthy() {
            let child = interp.child_scope(scope);
            return then_body(interp, child, depth);
        }
        for (ec, eb) in &elifs {
            if ec(interp, scope, depth).truthy() {
                let child = interp.child_scope(scope);
                return eb(interp, child, depth);
            }
        }
        if let Some(ob) = &orelse {
            let child = interp.child_scope(scope);
            return ob(interp, child, depth);
        }
        Flow::Normal
    })
}
