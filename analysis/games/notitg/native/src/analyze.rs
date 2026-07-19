//! Static body analysis for the snapshot optimization: the set of global names
//! the body could resolve to a DATA TABLE and NEVER writes. Only such names are
//! safe to snapshot into a native table (a written table's later value would
//! diverge from the frozen copy). Conservative: a name assigned ANYWHERE in the
//! body - as a bare target, an index target `n[..] =`, or a field target `n.x =`
//! - is excluded, as is any `local` of that name (a local shadows the global).
//!
//! The result is a candidate set; the frontier still decides at snapshot time
//! whether a candidate actually IS a host array table (else it stays crossing).

use std::collections::HashSet;

use crate::ast::{Expr, Stmt};

/// Names that are safe to snapshot: referenced but never written / shadowed.
pub fn snapshottable_names(body: &[Stmt]) -> HashSet<String> {
    let mut refs = HashSet::new();
    let mut written = HashSet::new();
    scan_stmts(body, &mut refs_and_written(&mut refs, &mut written));
    refs.retain(|n| !written.contains(n));
    refs
}

/// A tiny visitor state: collect referenced names and written/shadowed names.
struct Collect<'a> {
    refs: &'a mut HashSet<String>,
    written: &'a mut HashSet<String>,
}

fn refs_and_written<'a>(
    refs: &'a mut HashSet<String>,
    written: &'a mut HashSet<String>,
) -> Collect<'a> {
    Collect { refs, written }
}

fn scan_stmts(body: &[Stmt], c: &mut Collect) {
    for stmt in body {
        scan_stmt(stmt, c);
    }
}

fn scan_stmt(stmt: &Stmt, c: &mut Collect) {
    match stmt {
        Stmt::Assign { targets, values } => {
            for t in targets {
                mark_target(t, c);
            }
            for v in values {
                scan_expr(v, c);
            }
        }
        Stmt::Local { names, values } => {
            // A local shadows the global for the rest of the scope - exclude it.
            for n in names {
                c.written.insert(n.to_string());
            }
            for v in values {
                scan_expr(v, c);
            }
        }
        Stmt::If {
            cond,
            body,
            elifs,
            orelse,
        } => {
            scan_expr(cond, c);
            scan_stmts(body, c);
            for (ec, eb) in elifs {
                scan_expr(ec, c);
                scan_stmts(eb, c);
            }
            scan_stmts(orelse, c);
        }
        Stmt::NumericFor {
            var,
            start,
            stop,
            step,
            body,
        } => {
            c.written.insert(var.to_string());
            scan_expr(start, c);
            scan_expr(stop, c);
            if let Some(s) = step {
                scan_expr(s, c);
            }
            scan_stmts(body, c);
        }
        Stmt::GenericFor {
            names,
            exprs,
            body,
        } => {
            for n in names {
                c.written.insert(n.to_string());
            }
            for e in exprs {
                scan_expr(e, c);
            }
            scan_stmts(body, c);
        }
        Stmt::While { cond, body } => {
            scan_expr(cond, c);
            scan_stmts(body, c);
        }
        Stmt::FuncDef {
            name,
            params,
            body,
        } => {
            c.written.insert(name.to_string());
            for p in params {
                c.written.insert(p.to_string());
            }
            scan_stmts(body, c);
        }
        Stmt::Return { values } => {
            for v in values {
                scan_expr(v, c);
            }
        }
        Stmt::ExprStmt(e) => scan_expr(e, c),
        Stmt::Unparsed => {}
    }
}

/// An assignment TARGET writes (or shadows) the base name: `n = ..`, `n[..] = ..`,
/// `n.x = ..` all make `n` unsafe to snapshot.
fn mark_target(target: &Expr, c: &mut Collect) {
    match target {
        Expr::Sym(name) => {
            c.written.insert(name.to_string());
        }
        Expr::Index { base, key } => {
            mark_base(base, c);
            scan_expr(key, c);
        }
        Expr::Field { base, .. } => mark_base(base, c),
        _ => {}
    }
}

fn mark_base(base: &Expr, c: &mut Collect) {
    match base {
        Expr::Sym(name) => {
            c.written.insert(name.to_string());
        }
        // `a.b.c = ..` / `a[i][j] = ..` writes the root `a`.
        Expr::Index { base, .. } | Expr::Field { base, .. } => mark_base(base, c),
        _ => scan_expr(base, c),
    }
}

fn scan_expr(expr: &Expr, c: &mut Collect) {
    match expr {
        Expr::Sym(name) => {
            c.refs.insert(name.to_string());
        }
        Expr::Index { base, key } => {
            scan_expr(base, c);
            scan_expr(key, c);
        }
        Expr::Field { base, .. } => scan_expr(base, c),
        Expr::Unary { operand, .. } => scan_expr(operand, c),
        Expr::Binary { left, right, .. } => {
            scan_expr(left, c);
            scan_expr(right, c);
        }
        Expr::Call { fn_, args } => {
            scan_expr(fn_, c);
            for a in args {
                scan_expr(a, c);
            }
        }
        Expr::Method { recv, args, .. } => {
            scan_expr(recv, c);
            for a in args {
                scan_expr(a, c);
            }
        }
        Expr::Table { array, fields } => {
            for v in array {
                scan_expr(v, c);
            }
            for (_, v) in fields {
                scan_expr(v, c);
            }
        }
        Expr::FuncExpr { body, .. } => scan_stmts(body, c),
        _ => {}
    }
}
